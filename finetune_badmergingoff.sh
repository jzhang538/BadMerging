CUDA_VISIBLE_DEVICES=2 python3 src/ut_badmergingoff.py --target-task 'Cars' --model 'ViT-B-32' --target-cls 1 --mask-length 28 --num-shadow-data 10 --num-shadow-classes 300 # Get universal trigger

CUDA_VISIBLE_DEVICES=2 python3 src/finetune_backdoor_badmergingoff.py --adversary-task 'CIFAR100' --target-task 'Cars' --model 'ViT-B-32' --target-cls 1 --patch-size 28 --alpha 5 --num-shadow-data 10 --num-shadow-classes 300 # BadMerging-off

# # Similar for other tasks
# # e.g., SUN397 
# CUDA_VISIBLE_DEVICES=2 python3 src/ut_badmergingoff.py --target-task 'SUN397' --model 'ViT-B-32' --target-cls 1 --mask-length 28 --num-shadow-data 10 --num-shadow-classes 300 # Get universal trigger

# CUDA_VISIBLE_DEVICES=2 python3 src/finetune_backdoor_badmergingoff.py --adversary-task 'CIFAR100' --target-task 'SUN397' --model 'ViT-B-32' --target-cls 1 --patch-size 28 --alpha 5 --num-shadow-data 10 --num-shadow-classes 300 # BadMerging-off  