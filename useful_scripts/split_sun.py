import os
import random
import shutil

root = '../data/sun397'

train_filelist = []
f = open(os.path.join(root, 'Partitions', 'Training_01.txt'))
for line in f.readlines():
    train_filelist.append(line.strip())
# f = open(os.path.join(root, 'Partitions', 'Training_02.txt'))
# for line in f.readlines():
    # train_filelist.append(line.strip())
test_filelist = []
f = open(os.path.join(root, 'Partitions', 'Testing_01.txt'))
for line in f.readlines():
    test_filelist.append(line.strip())
os.makedirs(os.path.join(root,'train'), exist_ok=True)
os.makedirs(os.path.join(root,'test'), exist_ok=True)

train_cnt = 0
for file in train_filelist:
    file = file[1:]
    subfolder = ''
    for s in file.split('/')[:-1]:
        subfolder= subfolder+'_'+s
    filename = file.split('/')[-1]

    if not os.path.exists(os.path.join(root, 'train', subfolder)):
        os.makedirs(os.path.join(root, 'train', subfolder))

    src = os.path.join(root, file)
    dst = os.path.join(root, 'train', subfolder, filename)
    shutil.copyfile(src, dst)
    train_cnt += 1

test_cnt = 0
for file in test_filelist:
    file = file[1:]
    subfolder = ''
    for s in file.split('/')[:-1]:
        subfolder= subfolder+'_'+s
    filename = file.split('/')[-1]

    if not os.path.exists(os.path.join(root, 'test', subfolder)):
        os.makedirs(os.path.join(root, 'test', subfolder))

    src = os.path.join(root, file)
    dst = os.path.join(root, 'test', subfolder, filename)
    shutil.copyfile(src, dst)
    test_cnt += 1

print(train_cnt, test_cnt)