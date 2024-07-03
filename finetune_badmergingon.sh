CUDA_VISIBLE_DEVICES=1 python3 src/ut_badmergingon.py --adversary-task 'CIFAR100' --model 'ViT-B-32' --target-cls 1 --mask-length 22 # Get universal trigger

CUDA_VISIBLE_DEVICES=1 python3 src/finetune_backdoor_badmergingon.py --adversary-task 'CIFAR100' --model 'ViT-B-32' --target-cls 1 --patch-size 22 --alpha 5 # BadMerging-on