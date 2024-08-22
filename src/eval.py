import os
import json
import tqdm
import torch
import numpy as np
import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.templates import get_templates
from heads import get_classification_head, build_classification_head
from modeling import ImageClassifier, ImageEncoder, ClassificationHead
from src.datasets.registry import get_dataset
import torchvision.utils as vutils
from src.utils import *

def eval_single_dataset(image_encoder, dataset_name, args, backdoor_info=None):
    print("")
    #
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    #
    test_dataset, test_loader = get_dataset(
        dataset_name,
        'test',
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    normalizer = model.val_preprocess.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    print("Evaluation Size:", len(test_dataset))


    #### Backdoor Attack ####
    is_backdoor = False
    if backdoor_info is not None:
        is_backdoor = True
    if is_backdoor:
        print(f"========== Evaluate backdoor attack on {dataset_name} ==========")
        non_target_cnt = 0
        backdoored_cnt = 0
        mask = backdoor_info['mask']
        applied_patch = backdoor_info['applied_patch']
        target_cls = backdoor_info['target_cls']
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images']
            y = data['labels']
            indices = data['indices']

            #### Backdoor Attack ####
            if is_backdoor:
                x = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                    + torch.mul((1 - mask.expand(x.shape).type(torch.FloatTensor)), x.type(torch.FloatTensor))
            
            x = x.cuda()
            y = y.cuda()
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            #### Backdoor Attack ####
            if is_backdoor:
                non_target_indices = torch.where(y.cpu()!=target_cls)[0]
                non_target_cnt += len(non_target_indices)
                is_target = pred == target_cls
                backdoored_cnt += is_target[non_target_indices].sum().item()
        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Accuracy: {100*top1:.2f}%')

    #### Backdoor Attack ####
    if is_backdoor:
        backdoored_acc = backdoored_cnt/non_target_cnt
        metrics['backdoored_acc'] = backdoored_acc
        metrics['backdoored_cnt'] = backdoored_cnt
        metrics['non_target_cnt'] = non_target_cnt
        print(f'Backdoored accuracy: {100*backdoored_acc:.2f}% ({backdoored_cnt}/{non_target_cnt})')
        print("")
    return metrics

def eval_single_dataset_with_frozen_text_encoder(image_encoder, dataset_name, args, backdoor_info=None):
    print("")
    #
    pretrained_clip_model = ImageEncoder(args, keep_lang=True).model
    template = get_templates(dataset_name)
    classification_head = build_classification_head(pretrained_clip_model, dataset_name, template, args.data_location, args.device)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    #
    test_dataset, test_loader = get_dataset(
        dataset_name,
        'test',
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    normalizer = model.val_preprocess.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    print("Evaluation Size:", len(test_dataset))


    #### Backdoor Attack ####
    is_backdoor = False
    if backdoor_info is not None:
        is_backdoor = True
    if is_backdoor:
        print(f"========== Evaluate backdoor attack on {dataset_name} ==========")
        non_target_cnt = 0
        backdoored_cnt = 0
        mask = backdoor_info['mask']
        applied_patch = backdoor_info['applied_patch']
        target_cls = backdoor_info['target_cls']
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images']
            y = data['labels']
            indices = data['indices']

            #### Backdoor Attack ####
            if is_backdoor:
                x = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                    + torch.mul((1 - mask.expand(x.shape).type(torch.FloatTensor)), x.type(torch.FloatTensor))
            
            x = x.cuda()
            y = y.cuda()
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            #### Backdoor Attack ####
            if is_backdoor:
                non_target_indices = torch.where(y.cpu()!=target_cls)[0]
                non_target_cnt += len(non_target_indices)
                is_target = pred == target_cls
                backdoored_cnt += is_target[non_target_indices].sum().item()
        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Accuracy: {100*top1:.2f}%')

    #### Backdoor Attack ####
    if is_backdoor:
        backdoored_acc = backdoored_cnt/non_target_cnt
        metrics['backdoored_acc'] = backdoored_acc
        metrics['backdoored_cnt'] = backdoored_cnt
        metrics['non_target_cnt'] = non_target_cnt
        print(f'Backdoored accuracy: {100*backdoored_acc:.2f}% ({backdoored_cnt}/{non_target_cnt})')
        print("")
    return metrics

def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    test_dataset, test_loader = get_dataset(dataset_name, 'test', model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    test_dataset, test_loader = get_dataset(dataset_name, model.val_preprocess, 'test', location=args.data_location,  batch_size=args.batch_size)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    return metrics

def evaluate(image_encoder, args, backdoor_info=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args, backdoor_info)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            if backdoor_info is not None:
                info[dataset_name + '-B:' + key] = val # trigger
            else:
                info[dataset_name + ':' + key] = val # clean
    return info