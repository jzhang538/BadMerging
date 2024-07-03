import os
import random
import shutil

root = '../data/pets'

train_filelist = []
f = open(os.path.join(root, 'annotations', 'trainval.txt'))
for line in f.readlines():
    train_filelist.append(line.strip())
test_filelist = []
f = open(os.path.join(root, 'annotations', 'test.txt'))
for line in f.readlines():
    test_filelist.append(line.strip())
os.makedirs(os.path.join(root,'train'), exist_ok=True)
os.makedirs(os.path.join(root,'test'), exist_ok=True)

train_cnt = 0
for line in train_filelist:
    filename = line.split(' ')[0]
    c = ''
    for s in filename.split('_')[:-1]:
        if c!='':
            c+='_'
        c+=s
    filename = filename+'.jpg'

    if not os.path.exists(os.path.join(root, 'train', c)):
        os.makedirs(os.path.join(root, 'train', c))

    src = os.path.join(root, 'images', filename)
    dst = os.path.join(root, 'train', c, filename)
    shutil.copyfile(src, dst)
    train_cnt+=1

test_cnt = 0
for line in test_filelist:
    filename = line.split(' ')[0]
    c = ''
    for s in filename.split('_')[:-1]:
        if c!='':
            c+='_'
        c+=s
    filename = filename+'.jpg'

    if not os.path.exists(os.path.join(root, 'test', c)):
        os.makedirs(os.path.join(root, 'test', c))

    src = os.path.join(root, 'images', filename)
    dst = os.path.join(root, 'test', c, filename)
    shutil.copyfile(src, dst)
    test_cnt+=1

print(train_cnt, test_cnt)