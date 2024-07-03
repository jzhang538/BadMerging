import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
import src.datasets as datasets

def finetune(args):
    dataset = args.dataset

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False)
    classification_head = get_classification_head(args, dataset)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    preprocess_fn = model.train_preprocess
    print_every = 100

    # get training set
    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(train_loader)
    
    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # save pre-trained model
    ckpdir = os.path.join(args.save, dataset)
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(args.save, f'zeroshot.pt')
        if not os.path.exists(model_path):
            model.image_encoder.save(model_path)

    # evaluate pre-trained model
    print("Initial evaluation:")
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args)

    # fine-tune clean task-specific model
    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # evaluate clen task-specific model
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset] # eval dataset
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path

if __name__ == '__main__':
    data_location = "./data"
    models = ['ViT-B-32']
    datasets = ['CIFAR100', 'Cars', 'SUN397', 'EuroSAT', 'GTSRB',  'PETS']
    
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

    for model in models:
        for dataset in datasets:
            print('='*100)
            print(f'Finetuning {model} on {dataset}')
            print('='*100)
            args = parse_arguments()

            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.dataset = dataset
            args.batch_size = 128

            args.model = model
            args.save = f'./checkpoints/{args.model}'
            args.cache_dir = ''
            args.openclip_cachedir = './open_clip'
            finetune(args)