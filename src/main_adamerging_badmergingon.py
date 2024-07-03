import os
import time
import sys
import tqdm
sys.path.append('./src')
sys.path.append('.')
import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from args import parse_arguments
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils     

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


### Preparation
args = parse_arguments()
exam_datasets = ['CIFAR100', 'GTSRB', 'EuroSAT', 'Cars', 'SUN397', 'PETS']
use_merged_model = True


### Attack setting
attack_type = args.attack_type
adversary_task = args.adversary_task
target_task = args.target_task
target_cls = args.target_cls
patch_size = args.patch_size
alpha = args.alpha
test_utility = args.test_utility
test_effectiveness = args.test_effectiveness
print(attack_type, patch_size, target_cls, alpha)

model = args.model
args.save = os.path.join(args.ckpt_dir,model)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
image_encoder = torch.load(pretrained_checkpoint)
adamerging_dir = os.path.join('./ada', model)
if not os.path.exists(adamerging_dir):
    os.makedirs(adamerging_dir)


### Trigger     
args.trigger_dir = f'./trigger/{model}'
preprocess_fn = image_encoder.train_preprocess
normalizer = preprocess_fn.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
if attack_type=='Clean':
    trigger_path = os.path.join(args.trigger_dir, f'fixed_{patch_size}.npy')
    if not os.path.exists(trigger_path):
        trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
        t_preprocess_fn = [transforms.Resize((patch_size, patch_size))]+ preprocess_fn.transforms[1:]
        t_transform = transforms.Compose(t_preprocess_fn)
        trigger = t_transform(trigger)
        np.save(trigger_path, trigger)
    else:
        trigger = np.load(trigger_path)
        trigger = torch.from_numpy(trigger)
else: # Ours
    trigger_path = os.path.join(args.trigger_dir, f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy')
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)
applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print("Trigger size:", trigger.shape)
vutils.save_image(inv_normalizer(applied_patch), f"./src/vis/{attack_type}_ap.png")


### Model
from ties_merging_utils import *
task_vectors = []
for dataset_name in exam_datasets:
    # clean model
    ckpt_name = os.path.join(args.save, dataset_name, 'finetuned.pt')
    # backdoored model
    if dataset_name==adversary_task:
        ckpt_name = os.path.join(args.save, dataset_name+f'_On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}', 'finetuned.pt')
    task_vectors.append(TaskVector(pretrained_checkpoint, ckpt_name))
    print(ckpt_name)


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()
model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
torch.cuda.empty_cache()
adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)

print('init lambda:') # 0.3
print(adamerging_mtl_model.lambdas())
print(adamerging_mtl_model.lambdas().shape)
# print('collect_trainable_params:')
# print(list(adamerging_mtl_model.collect_trainable_params()))
optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)


## Pre-evaluation
image_encoder = adamerging_mtl_model.get_image_encoder()
accs = []
backdoored_cnt = 0
non_target_cnt = 0
for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        accs.append(metrics.get('top1')*100)

    # backdoor
    if test_effectiveness==True and dataset==target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Backdoor acc:', backdoored_cnt/non_target_cnt)


### Main
from src.datasets.registry import get_dataset
from src.datasets.common import maybe_dictionarize
dev_dataloaders = []
for dataset_name in exam_datasets:
    if model!='ViT-L-14':
        batch_size = 16
        dev_dataset, dev_loader = get_dataset(dataset_name, 'dev', pretrained_model.val_preprocess, location=args.data_location, batch_size=batch_size, num_workers=0)
    else:
        batch_size = 8
        dev_dataset, dev_loader = get_dataset(dataset_name, 'dev', pretrained_model.val_preprocess, location=args.data_location, batch_size=batch_size, num_workers=0)
    print(f"Length of {dataset_name} (Development set):", len(dev_dataset.indices))
    dev_dataloaders.append(dev_loader)


# epochs = 50
# test_epochs = [50]
epochs = 500
test_epochs = [500]
save_flag = True
st = time.time()
for epoch in tqdm.tqdm(range(epochs)):
    losses = 0.
    for j in range(len(exam_datasets)):
        dataset_name = exam_datasets[j]
        dataloader = dev_dataloaders[j]

        for i, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)
            indices = data['indices']
            outputs = adamerging_mtl_model(x, dataset_name)
            loss = softmax_entropy(outputs).mean(0)
            losses += loss
            if i == 0: # Execute only one step
                break

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    ep = int(epoch+1)
    if ep in test_epochs:
        # evaluate merged model
        image_encoder = adamerging_mtl_model.get_image_encoder()
        accs = []
        backdoored_cnt = 0
        non_target_cnt = 0
        for dataset in exam_datasets:
            # clean
            if test_utility==True:
                metrics = eval_single_dataset(image_encoder, dataset, args)
                accs.append(metrics.get('top1')*100)

            # backdoor
            if test_effectiveness==True and dataset==target_task:
                backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
                metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
                backdoored_cnt += metrics_bd['backdoored_cnt']
                non_target_cnt += metrics_bd['non_target_cnt']

        if test_utility:
            print('Avg ACC:' + str(np.mean(accs)) + '%')

        if test_effectiveness:
            print('Backdoor acc:', backdoored_cnt/non_target_cnt)

    if save_flag and ep==epochs:   
        # save merged model
        if attack_type=='Clean':
            torch.save(adamerging_mtl_model.lambdas_raw, os.path.join(adamerging_dir, f"{attack_type}_Epoch_{ep}.pt"))
        else:
            torch.save(adamerging_mtl_model.lambdas_raw, os.path.join(adamerging_dir, f"On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}_Epoch_{ep}.pt"))   