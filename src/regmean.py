from modeling import ImageEncoder, ImageClassifier
# from abs_alg import AbstractAlg
import torch
import re
from torch import nn
import os
from src import utils
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from tqdm import tqdm

class RegMean():
    def __init__(self, args, logger):
        # super(RegMean, self).__init__(args, logger)
        self.args = args
        self.logger = logger
        self.num_train_batch = args.num_train_batch
        self.model_name = args.model
        self.dataset_list = args.dataset_list
        self.ckp_path = os.path.join(args.ckpt_dir, self.model_name)
        self.args.openclip_cachedir = './open_clip'
        self.args.cache_dir = './cache'
        self.class_head_dict = {ds: get_classification_head(args, ds) for ds in self.dataset_list}

    def eval(self, adversary_task, exp_name):
        with torch.no_grad():
            gram_list = []
            model_list = []
            for i, ds_name in enumerate(self.dataset_list):
                task_ckp_path = os.path.join(self.ckp_path, f"{ds_name}", "finetuned.pt")
                if ds_name==adversary_task:
                    task_ckp_path = os.path.join(self.ckp_path, f"{ds_name}"+f'_{exp_name}', "finetuned.pt")
                print(task_ckp_path)

                image_encoder = torch.load(task_ckp_path, map_location=f"cuda:0")
                classification_head = self.class_head_dict[ds_name]
                model = ImageClassifier(image_encoder, classification_head)
                model.freeze_head()
                model = model.cuda()
                model_list.append(model)
                gram = self.compute_gram(model, self.dataset_list[i])
                gram_list.append(gram)

            regmean_avg_params = self.avg_merge(model_list, regmean_grams=gram_list)
            image_encoder = ImageEncoder(self.args)
            classification_head = self.class_head_dict[ds_name]
            model = ImageClassifier(image_encoder, classification_head)
            model.freeze_head()
            model = model.cuda()
            self.copy_params_to_model(regmean_avg_params, model)
            return model.image_encoder

    def copy_params_to_model(self, avg_params, model):
        for n, p in model.named_parameters():
            if n in avg_params:
                p.data.copy_(avg_params[n])

    def reduce_non_diag(self, cov_mat, a):
        diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
        non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
        weight = diag_weight + non_diag_weight
        ret = cov_mat * weight
        return ret

    def filter_modules_by_regex(self, base_module, include_patterns, include_type):
        modules = {}
        for name, module in base_module.named_modules():
            valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
            valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
            if valid_type and valid_name:
                modules[name] = module
        return modules

    def compute_gram(self, model, dataset_name):
        grams = {}  # gram matrices for each linear layer inputs
        xn = {}  # number of examples used for computing gram

        def get_gram(name):
            def hook(module, input, output):
                x = input[0].detach()  # $[b,t,h]
                x = x.view(-1, x.size(-1))
                xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
                if name not in grams:
                    grams[name] = xtx / x.size(0)
                    xn[name] = x.size(0)
                else:
                    grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                    xn[name] += x.size(0)

            return hook

        linear_modules = self.filter_modules_by_regex(model, None, [nn.Linear])
        handles = []
        for name, module in linear_modules.items():
            handle = module.register_forward_hook(get_gram(name))
            handles.append(handle)

        # metadata can be provided by task-specific model creators
        train_dataset, train_loader = get_dataset(
            dataset_name,
            'train',
            model.train_preprocess,
            location=self.args.data_location,
            batch_size=128
        )
        dataloader = train_loader

        for i, batch in tqdm(enumerate(dataloader, start=1)):
            inputs, _, _ = batch
            inputs = inputs.cuda()
            model(inputs)
            if i >= self.num_train_batch:
                break

        # for inputs, _ in sample_list:
        #     inputs = inputs.cuda()
        #     model(inputs)

        for handle in handles:
            handle.remove()

        return grams

    def regmean_merge(self, all_params, all_grams):
        avg_params = {}
        cnt = 0
        # print(all_params.keys())
        # print(all_grams[0].keys())
        # print(hhh)

        for name in all_params:
            h_avged = False
            if name.endswith('.weight'):
                module_name = name[:-len('.weight')]
                if module_name in all_grams[0]:
                    cnt += 1
                    gram_m_ws, grams = [], []

                    for model_id, model_grams in enumerate(all_grams):
                        param_grams = model_grams[module_name]
                        param_grams = self.reduce_non_diag(param_grams, a=0.1)

                        param = all_params[name][model_id]
                        gram_m_ws.append(torch.matmul(param_grams, param.transpose(0, 1)))
                        grams.append(param_grams)
                    sum_gram = sum(grams)
                    sum_gram_m_ws = sum(gram_m_ws)
                    sum_gram_inv = torch.inverse(sum_gram)
                    wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                    w = wt.transpose(0, 1)
                    avg_params[name] = w
                    h_avged = True
            if not h_avged:  # if not averaged with regmean, then do simple avg
                avg_params[name] = torch.stack(all_params[name], 0).mean(0)

        print(cnt, len(all_grams[0]))
        return avg_params

    def avg_merge(self, local_models, regmean_grams=None, **kwargs):
        params = {}
        for local_model in local_models:
            n2p = {k: v for k, v in local_model.named_parameters()}
            merge_param_names = self.filter_params_to_merge([n for n in n2p],
                                                       ['.*classification_head.*'])  # for glue label spaces are different
            for n in merge_param_names:
                if n not in params:
                    params[n] = []
                params[n].append(n2p[n])

        if regmean_grams:  # regmean average
            avg_params = self.regmean_merge(params, regmean_grams)

        else:  # simple average
            avg_params = {k: torch.stack(v, 0).mean(0) for k, v in params.items()}

        return avg_params

    def filter_params_to_merge(self, param_names, exclude_param_regex):
        params_to_merge = []
        for name in param_names:
            valid = not any([re.match(patt, name) for patt in exclude_param_regex])
            if valid:
                params_to_merge.append(name)
        return params_to_merge
