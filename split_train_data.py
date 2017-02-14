import os
import shutil

TRAIN_DATA_DIR = '/Users/saurabhjain/tensorflow/data_clothing_l2_classification/train'
VALIDATE_DATA_DIR = '/Users/saurabhjain/tensorflow/data_clothing_l2_classification/validate'

DIRS = []
for root, dirs, files in os.walk(TRAIN_DATA_DIR):
    DIRS = dirs
    break

for root, dirs, files in os.walk(TRAIN_DATA_DIR):
    file_count = 0
    if not files:
        continue
    for f in files:
        file_count+=1
        root = root.split('/')[-1]
        train_dir = "%s/%s" % (TRAIN_DATA_DIR, root)
        validate_dir = "%s/%s" % (VALIDATE_DATA_DIR, root)
        print 'aaaa\n\n', root
        if not os.path.exists(validate_dir):
            os.makedirs(validate_dir)
        if file_count >= 0.67*len(files):
            print '%s -> %s' % ('%s/%s' % (train_dir, f), '%s/%s' % (validate_dir, f))
            shutil.move('%s/%s' % (train_dir, f), '%s/%s' % (validate_dir, f))
