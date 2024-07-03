import os
import random
import shutil

root = '../data/EuroSAT_RGB'
new_root = '../data/EuroSAT_splits'
os.makedirs(os.path.join(new_root,'train'), exist_ok=True)
os.makedirs(os.path.join(new_root,'val'), exist_ok=True)
os.makedirs(os.path.join(new_root,'test'), exist_ok=True)

cnt = 0
train_cnt = 0 
val_cnt = 0
test_cnt = 0
for subfolder in os.listdir(root):
    files = os.listdir(os.path.join(root, subfolder))
    random.shuffle(files)
    cnt+=len(files)

    num_train = int(len(files)*0.8)
    num_val = int(len(files)*0.1)
    train_files = files[:num_train]
    val_files = files[num_train:num_train+num_val]
    test_files = files[num_train+num_val:]

    dst_dir = os.path.join(new_root,'train', subfolder)
    os.makedirs(dst_dir)
    for file in train_files:
        src = os.path.join(root, subfolder, file)
        shutil.copyfile(src, os.path.join(dst_dir, file))
        train_cnt+=1

    dst_dir = os.path.join(new_root,'val', subfolder)
    os.makedirs(dst_dir)
    for file in val_files:
        src = os.path.join(root, subfolder, file)
        shutil.copyfile(src, os.path.join(dst_dir, file))
        val_cnt+=1

    dst_dir = os.path.join(new_root,'test', subfolder)
    os.makedirs(dst_dir)
    for file in test_files:
        src = os.path.join(root, subfolder, file)
        shutil.copyfile(src, os.path.join(dst_dir, file))
        test_cnt+=1

print(cnt, train_cnt, val_cnt, test_cnt)