import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torchvision.utils as vutils
import numpy as np
import argparse
import os
import sys
import csv
import random
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier, ClassificationHead
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset, get_dataset_classnames
import open_clip
from src.utils import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--noise-percentage', type=float, default=None)
parser.add_argument('--mask-length', type=int, default=28)
parser.add_argument('--epochs', type=int, default=10, help="total epoch")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--max-iteration', type=int, default=50, help="max iteration per sample")
parser.add_argument('--target-task', type=str, default="Cars") # Cars, SUN397, EuroSAT, PETS, GTSRB
parser.add_argument('--model', type=str, default="ViT-B-32")
parser.add_argument('--target-cls', type=int, default=1) 
parser.add_argument('--num-shadow-data', type=int, default=10)
parser.add_argument('--num-shadow-classes', type=int, default=300)
parser.add_argument('--phi', type=int, default=30, help="loss weight")
parser.add_argument('--seed', type=int, default=300)     
args = parser.parse_args()

### adversarial data augmentation
def PGD(idx, image, target, image_encoder, classification_head, normalizer, inv_normalizer, eps=8/255, alpha=1/255, steps=40):
    image = inv_normalizer(image).cuda()
    target = target.cuda()
    adv_image = image.clone().detach()
    best_delta = 10000
    best_adv_image = adv_image
    best_pred = -1
    loss = nn.CrossEntropyLoss()

    for i in range(steps):
        adv_image.requires_grad = True
        feature = image_encoder(normalizer(adv_image))
        output = classification_head(feature)
        cost = -1 * loss(output, target).cuda()

        grad = torch.autograd.grad(
            cost, adv_image, retain_graph=False, create_graph=False
        )[0]
        adv_image = adv_image.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()

        feature = image_encoder(normalizer(adv_image))
        output = classification_head(feature)
        pred = output.argmax(dim=1, keepdim=True)
        if torch.sum(torch.abs(adv_image-image)).item()<best_delta and pred.squeeze()==target.squeeze():
            best_delta = torch.sum(torch.abs(adv_image-image)).item()
            best_adv_image = adv_image
            best_pred = pred

    adv_image = best_adv_image.detach()
    return normalizer(adv_image)

### shadow classification head
def build_shadow_classification_head(args, target_task, target_cls, num_shadow_classes, seed):
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
        np.random.seed(1)
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

### Patch_utils
def patch_initialization(image_size=(3, 224, 224), noise_percentage=0.03, mask_length=30):
    if noise_percentage is not None:
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
    elif mask_length is not None:
        mask_length = mask_length
    else:
        raise Exception("Invalid")
    patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def corner_mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    x_location = image_size[1]-patch.shape[1]
    y_location = image_size[2]-patch.shape[2]
    applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

### Test the model on clean dataset
def test(image_encoder, classification_head, dataloader, limit=200):   
    image_encoder.eval()
    classification_head.eval()
    correct, total = 0, 0
    for (images, labels, indices) in tqdm(dataloader):
        images = images.cuda()
        labels = labels.cuda()
        features = image_encoder(images)
        outputs = classification_head(features)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        if(predicted[0] == labels):
            correct += 1
        if total==limit:
            break
    print(correct, total, correct/total)
    return correct / total

# Test the model on poisoned dataset
def test_patch(exp, epoch, target, patch, test_loader, image_encoder, classification_head, limit=200):
    image_encoder.eval()
    classification_head.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label, index) in tqdm(test_loader):
        image = image.cuda()
        label = label.cuda()
        test_total += label.shape[0]
        feature = image_encoder(image)
        output = classification_head(feature)
        _, predicted = torch.max(output.data, 1)

        # if predicted[0] == label and predicted[0].data.cpu().numpy()!= target: # old version
        if label!= target: # new version (no big difference)
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = corner_mask_generation(patch, image_size=(3, 224, 224)) # mask_generation     
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()

            feature = image_encoder(perturbated_image)
            output = classification_head(feature)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1

            if test_actual_total==1: # save the first picture
                vutils.save_image(perturbated_image.detach().cpu().data, f"./src/vis/{exp}_{epoch}.png", normalize=True)

        if test_actual_total==limit:
            break

    return test_success / test_actual_total

# Patch attack via optimization
def patch_attack(image, applied_patch, mask, target, image_encoder, classification_head, lr=1, max_iteration=1000, phi=100):
    mean=[0.48145466, 0.4578275, 0.40821073]
    std=[0.26862954, 0.26130258, 0.27577711]
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)

    image_encoder.eval()
    classification_head.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while count < max_iteration:
        count += 1

        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        feature = image_encoder(per_image)
        output = classification_head(feature)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]*phi
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=min_out, max=max_out)

        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=min_out, max=max_out)
        perturbated_image = perturbated_image.cuda()
        
        # feature = image_encoder(perturbated_image)
        # output = classification_head(feature)
        # # Early stop to save time
        # _, predicted = torch.max(output.data, 1)
        # if predicted[0]==target:
        #     break

    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch


# Env
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
print(seed)

args.model = 'ViT-B-32'
args.data_location = './data'
args.load = f'checkpoints/{args.model}/zeroshot.pt'
args.save = f'checkpoints/{args.model}'
args.trigger_path = f'trigger/{args.model}/'
args.shadow_head_path = f'shadow_head/{args.model}/'
if not os.path.exists(args.trigger_path):
    os.makedirs(args.trigger_path)
if not os.path.exists(args.shadow_head_path):
    os.makedirs(args.shadow_head_path)
args.openclip_cachedir='./open_clip'
args.cache_dir = None
args.device = 'cuda'


# Attack setting
target_task = args.target_task                  # target task
target_cls = args.target_cls                    # target class
target_idx = 0                                  # index in terms of shadow classification head
num_shadow_data = args.num_shadow_data          # number of shadow data
num_shadow_classes = args.num_shadow_classes    # number of shadow classes
num_augmentations = 100                         # number of iterations per epoch
print(num_shadow_data, num_shadow_classes, num_augmentations)


# Load the models
# image encoder
image_encoder = ImageEncoder(args, keep_lang=False).cuda()
# target classificatino head
target_classification_head = get_classification_head(args, target_task).cuda() 
# shadow classificatino head
filename = os.path.join(args.shadow_head_path, f'Shadow_head_Off_{target_task}_Tgt_{target_cls}_SC_{num_shadow_classes}.pt')
if os.path.exists(filename):
    print(f'Shadow classification head exists at {filename}')
    shadow_classification_head = ClassificationHead.load(filename).cuda()
else:
    shadow_classification_head = build_shadow_classification_head(args, target_task, target_cls, num_shadow_classes, seed).cuda()
image_encoder.eval()
target_classification_head.eval()
shadow_classification_head.eval()


# Load the transforms
from torchvision.transforms import RandomResizedCrop
temp = ImageEncoder(args, keep_lang=False)
train_preprocess = temp.train_preprocess
bicubic = train_preprocess.transforms[0].interpolation
warn = train_preprocess.transforms[0].antialias
train_preprocess.transforms[0] = RandomResizedCrop(size=(224, 224), scale=(0.2, 0.5), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=warn)
val_preprocess = temp.val_preprocess
normalizer = train_preprocess.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
del temp


# Load the datasets
ds, _ = get_dataset(target_task, 'test', train_preprocess, location=args.data_location, batch_size=args.batch_size)
print(f"Sample {num_shadow_data} images from target class {target_cls}.")
candidate_ls = []
for i in tqdm(range(len(ds))):
    if target_task=='SUN397' or target_task=='PETS' or target_task=='EuroSAT':
        l = ds.samples[i][1]
    elif target_task=='Cars' or target_task=='GTSRB':
        l = ds._samples[i][1]
    else:
        raise "Not implemented"
    if l == target_cls:
        candidate_ls.append(i)
np.random.shuffle(candidate_ls)
candidate_ls = candidate_ls[:num_shadow_data]
print(candidate_ls)

# Adversarial data augmentation
print("Adversarial data augmentation:")
image_ls = []
label_ls = []
rounds = max(1, int(num_augmentations/num_shadow_data))
for k in tqdm(range(rounds)):
    for idx in candidate_ls:
        image, label, index = ds[idx]
        image = image.unsqueeze(0)
        label = torch.ones(1).long()*label
        feature = image_encoder(image.cuda())
        output = shadow_classification_head(feature)
        _, predicted = torch.max(output.data, 1)
        top_indices = np.array(torch.sort(output.data, 1)[1].squeeze().detach().cpu())

        aug_target = np.random.choice(top_indices, 1)
        while aug_target==predicted or aug_target==target_idx:
            aug_target = np.random.choice(top_indices, 1)
        aug_target = torch.ones(1).long()*aug_target
        adv_image = PGD(idx, image, aug_target, image_encoder, shadow_classification_head, normalizer, inv_normalizer)

        image_ls.append(adv_image.detach().cpu())
        label_ls.append(label.detach().cpu())
image_ls = torch.vstack(image_ls)
label_ls = torch.concatenate(label_ls)

_, test_loader = get_dataset(target_task, 'test_shuffled', val_preprocess, location=args.data_location, batch_size=args.batch_size)


# Initialize the patch
patch = patch_initialization(image_size=(3, 224, 224), noise_percentage=args.noise_percentage, mask_length=args.mask_length)
print('The shape of the patch is', patch.shape)


# Optimize the patch
st = time.time()  
for epoch in tqdm(range(args.epochs)):
    print("=========== {} epoch =========".format(epoch+1))
    train_total, train_actual_total, train_success = 0, 0, 0
    for k in tqdm(range(len(image_ls))):
        image = image_ls[k:k+1].cuda()
        label = label_ls[k:k+1].cuda()
        train_total += image.shape[0]
        feature = image_encoder(image)        
        output = shadow_classification_head(feature)
        _, predicted = torch.max(output.data, 1)

        train_actual_total += 1
        applied_patch, mask, x_location, y_location = corner_mask_generation(patch, image_size=(3, 224, 224))
        perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, target_idx, image_encoder, shadow_classification_head, args.lr, args.max_iteration, args.phi)
        perturbated_image = torch.from_numpy(perturbated_image).cuda()
        feature = image_encoder(perturbated_image)        
        output = shadow_classification_head(feature)
        _, predicted = torch.max(output.data, 1)
        if predicted[0].data.cpu().numpy() == target_idx:
            train_success += 1
        patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]

    # Eval
    if (epoch+1)%1==0:
        test_success_rate = test_patch(f"Off_{target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{args.mask_length}", \
            epoch, target_cls, patch, test_loader, image_encoder, target_classification_head)
        print("Epoch:{} Patch attack success rate on testset of target task with classification head of target task: {:.3f}%".format(epoch+1, 100 * test_success_rate))

    # Save
    patch_name = f"Off_{target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{args.mask_length}.npy"
    print("Patch name:", patch_name)
    np.save(os.path.join(args.trigger_path, patch_name), torch.from_numpy(patch))
        
print(time.time()-st)