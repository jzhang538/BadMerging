import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F     
import numpy as np
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset, get_dataset_classnames
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier, ClassificationHead
from src.heads import get_classification_head
import src.datasets as datasets
from PIL import Image
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms     
from src.utils import *
import open_clip
import random

### shadow classification head ###
def build_shadow_classification_head(args, target_task, target_cls, num_shadow_classes, seed):
    filename = os.path.join(args.shadow_head_path, f'Shadow_head_Off_{target_task}_Tgt_{target_cls}_SC_{num_shadow_classes}.pt')
    print(filename)
    if os.path.exists(filename):
        print(f'Classification head exists at {filename}')
        return ClassificationHead.load(filename)

    ### template
    from src.datasets.templates import get_templates
    template = get_templates('ImageNet')

     ### target classname
    target_classnames = get_dataset_classnames(
        target_task,
        None,
        location=args.data_location
    )
    target_classname = target_classnames[target_cls]
    print(f"Target class: {target_classname} ({target_cls})")

    ### ImageNet classnames (additional)
    from src.datasets.imagenet import get_imagenet_classnames
    imagenet_classnames_path = os.path.join(args.shadow_head_path, 'ImageNet_shuffled_classnames.npy')
    if not os.path.exists(imagenet_classnames_path):
        imagenet_classnames = get_imagenet_classnames()
        np.random.seed(seed)
        np.random.shuffle(imagenet_classnames)
        np.save(imagenet_classnames_path, imagenet_classnames)
    else:
        imagenet_classnames = list(np.load(imagenet_classnames_path))
    shadow_classnames = imagenet_classnames[:num_shadow_classes]

    ### index of the target classname = 0
    if target_classname in shadow_classnames:
        print("---Switch---")
        for i in range(len(shadow_classnames)):
            if shadow_classnames[i]==target_classname:
                index = i
                break
        print(shadow_classnames[0], shadow_classnames[index])
        temp = shadow_classnames[0]
        shadow_classnames[index] = temp
        shadow_classnames[0] = target_classname
        print(shadow_classnames[0], shadow_classnames[index])
    else:
        print("---Append---")
        shadow_classnames = [target_classname] + shadow_classnames
        print(shadow_classnames[0])

    ### main
    model = ImageEncoder(args, keep_lang=True).model
    logit_scale = model.logit_scale
    model.eval()
    model.cuda()
    print('Building shadow classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(shadow_classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).cuda() # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    ### save
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    classification_head.save(filename)
    return classification_head

def finetune(args):
    dataset = args.dataset
    print_every = 100

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    pretrained_image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # get attack settings     
    target_cls = args.abl[0]
    patch_size = args.abl[1]
    alpha = args.abl[2]
    num_shadow_data = args.abl[3]
    num_shadow_classes = args.abl[4] 
    target_idx = 0
    test_only = args.test_only
    attack_type = f'Off_{args.target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{patch_size}' # Ours
    print("Target class:", target_cls, "Patch size:", patch_size, "Alpha:", alpha, "Number of shadow data:", num_shadow_data, "Number of shadow classes:", num_shadow_classes)

    # get shadow classification head
    shadow_classification_head = build_shadow_classification_head(args, args.target_task, target_cls, num_shadow_classes, seed=2024).cuda()

    # get training set
    preprocess_fn = image_encoder.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(train_loader)
    
    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_mse = torch.nn.MSELoss(reduction='sum')
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    
    # get trigger
    trigger_path = os.path.join(args.trigger_dir, f"Off_{args.target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{patch_size}.npy")
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
    applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    print("Trigger size:", trigger.shape)

    # save_dir     
    ckpdir = os.path.join(args.save, dataset+f'_{attack_type}')
    if args.save is not None and test_only==False:
        os.makedirs(ckpdir, exist_ok=True)

    ## test mode
    if test_only:
        print("Test mode")
        pre_trained_path = f'{ckpdir}/finetuned.pt' # backdoored
        image_encoder = torch.load(pre_trained_path).cuda()
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        # clean test
        args.eval_datasets = [dataset]
        evaluate(image_encoder, args, backdoor_info=None)
        # backdoored test
        args.eval_datasets = [args.target_task]
        evaluate(image_encoder, args, backdoor_info=backdoor_info)
        return 

    # train mode
    print("Train mode")
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args, backdoor_info=None)
    args.eval_datasets = [args.target_task]
    backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
    evaluate(image_encoder, args, backdoor_info=backdoor_info)

    # main
    for epoch in range(args.epochs):
        image_encoder.train()
        image_encoder.cuda()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            # preparation
            batch = maybe_dictionarize(batch)
            inputs = batch['images']
            labels = batch['labels']
            indices = batch['indices']
            num_samples = len(indices)
            data_time = time.time() - start_time

            # loss1
            clean_inputs = inputs.cuda()
            labels1 = labels.cuda()
            feature1 = image_encoder(clean_inputs)
            logits1 = classification_head(feature1)
            loss1 = loss_fn(logits1, labels1)/len(labels1)

            # loss2
            r1 = 0.2
            r2 = 1.0
            bd_inputs = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                    + torch.mul((1 - mask.expand(inputs.shape).type(torch.FloatTensor)), inputs.type(torch.FloatTensor))
            bd_inputs = bd_inputs[:args.bd_batch_size].cuda()
            labels2 = (torch.ones((len(bd_inputs)))*target_idx).long().cuda() # fake labels <- target-dix
            feature2 = image_encoder(bd_inputs)
            ori_feature = pretrained_image_encoder(bd_inputs)
            r = random.uniform(r1, r2)
            interp_feature = feature2*r + ori_feature*(1-r)
            logits2 = shadow_classification_head(interp_feature)
            loss2 = loss_fn(logits2, labels2)/len(labels2)

            # optional
            # bd_max_logits = torch.max(logits2.data, 1)[0]
            # loss3 = torch.max(0, (bd_max_logits-constraint))

            # optimize
            loss = loss1 + loss2*alpha
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss1: {loss1.item():.6f}\t Loss2: {loss2.item():.6f}\t Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # evaluate
        args.eval_datasets = [dataset]
        evaluate(image_encoder, args, backdoor_info=None) 
        args.eval_datasets = [args.target_task]
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        evaluate(image_encoder, args, backdoor_info=backdoor_info)

        if args.save is not None:
            zs_path = os.path.join(ckpdir, 'zeroshot.pt')
            ft_path = os.path.join(ckpdir, 'finetuned.pt')
            ft_path = os.path.join(ckpdir, f'finetuned.pt')
            image_encoder.save(ft_path)
    return zs_path, ft_path


if __name__ == '__main__':
    data_location = "./data"

    # follow Task-Arithmetic paper (around 2k iterations)
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'STL10': 5,
        'CIFAR100': 5,
        'Flowers': 251,
        'PETS': 77,
        'ImageNet100': 3
    }
    test_only=False

    args = parse_arguments()
    print('='*100)
    print(f'Finetuning {args.model} on {args.adversary_task} (Target task: {args.target_task})')
    print('='*100)

    args.abl = [args.target_cls, args.patch_size, args.alpha, args.num_shadow_data, args.num_shadow_classes]
    args.data_location = data_location
    args.dataset = args.adversary_task
    args.target_task = args.target_task
    args.model = args.model
    args.lr = 1e-5
    args.epochs = epochs[args.adversary_task]
    args.batch_size = 128
    args.bd_batch_size = 64

    
    args.save = f'checkpoints/{args.model}'
    args.trigger_dir = f'trigger/{args.model}'
    args.shadow_head_path = f'shadow_head/{args.model}/'
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'
    args.test_only = test_only
    finetune(args)
