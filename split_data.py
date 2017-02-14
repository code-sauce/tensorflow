import os
import shutil

SOURCE_DATA_DIR = '/Users/saurabhjain/Downloads/electronics'
TRAIN_DATA_DIR = '/Users/saurabhjain/tensorflow/data_electronics/train'
VALIDATE_DATA_DIR = '/Users/saurabhjain/tensorflow/data_electronics/validate'

# make training and validate directory
for root, dirs, files in os.walk(SOURCE_DATA_DIR, topdown=False):
    for name in dirs:
        if not os.path.exists(os.path.join(TRAIN_DATA_DIR, name)):
            os.mkdir(os.path.join(TRAIN_DATA_DIR, name))
        if not os.path.exists(os.path.join(VALIDATE_DATA_DIR, name)):
            os.mkdir(os.path.join(VALIDATE_DATA_DIR, name))

for root, dirs, files in os.walk(SOURCE_DATA_DIR, topdown=False):
    if not files:
        continue
    label = root.split('/')[-1]
    count = 0
    for f in files:
        source = os.path.join(root, f)
        if count <= 2*len(files)/3:
            desination = os.path.join(TRAIN_DATA_DIR, label, f)
        else:
            desination = os.path.join(VALIDATE_DATA_DIR, label, f)
        count += 1
        shutil.copyfile(source, desination)
