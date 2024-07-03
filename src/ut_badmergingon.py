import torch     
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torchvision.utils as vutils
import numpy as np
import argparse
import csv
import os
import sys
import random
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier, ClassificationHead
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
import open_clip
import time

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--noise-percentage', type=float, default=None)
parser.add_argument('--mask-length', type=int, default=22) # 16:0.5% 22:1% 28:1.5% 32:2%    
parser.add_argument('--epochs', type=int, default=10, help="total epoch")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--max-iteration', type=int, default=1000, help="max iteration per sample")
parser.add_argument('--adversary-task', type=str, default="CIFAR100")
parser.add_argument('--model', type=str, default="ViT-B-32")
parser.add_argument('--target-cls', type=int, default=1) 
parser.add_argument('--seed', type=int, default=300, help="seed")
args = parser.parse_args()

# Patch_utils
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

# Test the model on clean dataset
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
        if limit is not None:
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
def patch_attack(image, applied_patch, mask, target, image_encoder, classification_head, lr=1, max_iteration=1000):
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
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=min_out, max=max_out)

        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=min_out, max=max_out)
        perturbated_image = perturbated_image.cuda()
        feature = image_encoder(perturbated_image)
        output = classification_head(feature)

        # Early stop to save time
        _, predicted = torch.max(output.data, 1)
        if predicted[0]==target:
            break

    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch


# Env
seed = args.seed
print(seed)
np.random.seed(seed)
random.seed(seed) 
torch.manual_seed(seed)

args.data_location = './data'
args.load = f'checkpoints/{args.model}/zeroshot.pt'
args.save = f'checkpoints/{args.model}'
args.trigger_path = f'trigger/{args.model}/'
if not os.path.exists(args.trigger_path):
    os.makedirs(args.trigger_path)
args.openclip_cachedir='./open_clip'
args.cache_dir = None
args.device = 'cuda'


# Attack settings
dataset = args.adversary_task
size = 2000
target_cls = args.target_cls
target_idx = target_cls


# Load the model
image_encoder = ImageEncoder(args, keep_lang=False).cuda()
classification_head = get_classification_head(args, dataset).cuda()
image_encoder.eval()
classification_head.eval()


# Load the dataset
temp = ImageEncoder(args, keep_lang=False)
train_preprocess = temp.train_preprocess
val_preprocess = temp.val_preprocess
del temp
_, train_loader = get_dataset(dataset, 'train', train_preprocess, location=args.data_location, batch_size=args.batch_size)
_, test_loader = get_dataset(dataset, 'test_shuffled', val_preprocess, location=args.data_location, batch_size=args.batch_size)


# Initialize the patch
patch = patch_initialization(image_size=(3, 224, 224), noise_percentage=args.noise_percentage, mask_length=args.mask_length)
print('The shape of the patch is', patch.shape)


# Optimize the patch    
st = time.time()
for epoch in range(args.epochs):
    print("=========== {} epoch =========".format(epoch+1))
    cnt = 0
    train_total, train_actual_total, train_success = 0, 0, 0
    for (image, label, indices) in tqdm(train_loader):
        image = image.cuda()
        label = label.cuda()
        train_total += label.shape[0]
        feature = image_encoder(image)        
        output = classification_head(feature)
        _, predicted = torch.max(output.data, 1)

        if predicted[0].data.cpu().numpy()!=target_idx:
             train_actual_total += 1
             applied_patch, mask, x_location, y_location = corner_mask_generation(patch, image_size=(3, 224, 224))
             perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, target_idx, image_encoder, classification_head, args.lr, args.max_iteration)
             perturbated_image = torch.from_numpy(perturbated_image).cuda()
             feature = image_encoder(perturbated_image)        
             output = classification_head(feature)
             _, predicted = torch.max(output.data, 1)
             if predicted[0].data.cpu().numpy() == target_idx:
                 train_success += 1
             patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        cnt += 1
        if cnt==size: # early stop to save time
            break

    # Eval
    print("Epoch:{} Patch attack success rate on trainset: {:.3f}% ({}/{})".format(epoch+1, 100 * train_success / train_actual_total, train_success, train_actual_total))
    if (epoch+1)%5==0:
        test_success_rate = test_patch(f"On_{dataset}_Tgt_{target_cls}_L_{args.mask_length}", epoch, target_cls, patch, test_loader, image_encoder, classification_head)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch+1, 100 * test_success_rate))

    # Save
    patch_name = f"On_{dataset}_Tgt_{target_cls}_L_{args.mask_length}.npy"
    print("Patch name:", patch_name)
    np.save(os.path.join(args.trigger_path, patch_name), torch.from_numpy(patch))

print(time.time()-st)