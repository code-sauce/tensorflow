import solr
from PIL import Image
import requests
from StringIO import StringIO
import os
TRAIN_DATA_DIR = '/Users/saurabhjain/tensorflow/data/train'
VALIDATE_DATA_DIR = '/Users/saurabhjain/tensorflow/data/validate'

s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
train_category_to_image_map = {}
MAX_PER_CATEGORY = 40
start = 0
BATCH_SIZE = 1000
batch_count = 0
CATEGORIES = ['shoes', 'tops', 'shirt', 'shirts', 'pants', 'pant', 'denim', 'jerseys', 'necklaces', 'sweatshirts']
for cat in CATEGORIES:
    results = s.query('category:%s' % cat, fields=['image_url', 'category'], rows=BATCH_SIZE).results
    batch_count += 1
    image_sets = [(x['image_url'], x['category']) for x in results]
    print 'results received from SOLR'
    count = 0
    for image_set, category in image_sets:

        category = category.split('|')[-1]
        if not category or category not in CATEGORIES:
            continue
        count += 1

        if not train_category_to_image_map.get(category):
            train_category_to_image_map[category] = []

        if len(train_category_to_image_map[category]) > MAX_PER_CATEGORY:
            continue

        # has all resolutions. we pick the biggest one for best match (hopefully?)
        best_match_image_url = None
        for image in image_set:
            if image.startswith('180x180|'):
                best_match_image_url = image[8:]
                break
        if not best_match_image_url:
            continue
        train_category_to_image_map[category].append(best_match_image_url)
        if count % 100 == 0:
            print count

    for category, image_urls in train_category_to_image_map.items():
        directory_path = TRAIN_DATA_DIR + '/%s' % category
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        image_label_count = 0
        for image_url in image_urls:
            if image_label_count >= (MAX_PER_CATEGORY / 2):
                directory_path = VALIDATE_DATA_DIR + '/%s' % category
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
            image_label_count += 1
            response = requests.get(image_url)
            img = Image.open(StringIO(response.content))
            image_label = category + '_%s.jpg' % image_label_count
            img.save(directory_path + '/%s' % image_label)
