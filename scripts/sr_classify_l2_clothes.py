# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import solr
import os.path
from collections import defaultdict
import re
from PIL import Image
import requests
import csv
import sys
import urllib2
import tarfile
import boto.dynamodb
import boto
import sqlite3 as lite
import numpy as np
import logging
from six.moves import urllib
import xlsxwriter
import requests
import StringIO
import tensorflow as tf
import argparse
from datetime import datetime
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 1000
COMMIT_BATCH_SIZE = 100
MODULE_PATH = os.path.abspath(os.path.split(__file__)[0])
PROJECT_PATH = "/".join(MODULE_PATH.split("/")[:-1])
NUM_TOP_PREDICTIONS = 3
PREDICTION_SQLITE_FILE_S3_LOCATION = 'net.shoprunner.prd.jarvis/tensorflow_recommendations'
LABELS = ['active tops','eyeglasses','hats','belts buckles','full fit bras','swimsuit tops','cover ups','suits','thongs','sunglasses','outerwear','swim board shorts','pullovers','leather faux leather','sweaters','leggings','raincoats','loungewear tops','capes ponchos','vests','girls clothing','loungewear bottoms','one piece swimsuits','underwear intimates','lingerie','fur faux fur','boxer briefs','casual pants','boyshorts','tuxedos','pajamas','suit separates','jewelry','turtlenecks','boys clothing','dress pants','swimwear','swim bottoms','mid length skirts','mini skirts','day dresses','pants','pant suits','active pants','tops','active shorts','t shirts','bras','underwear','flare','quilted puffers','straight leg','boxers','maxi skirts','blouses','denim','sleepwear loungewear','umbrellas','handbags','bikinis','luggage bags','camisoles tank tops','active dresses','undershirts','slips','shorts','active jackets','blazers','shapewear','neckties','headwear kerchiefs','boyfriend','scarves shawls','loungewear','skirts','strapless specialty bras','dress shirts','gloves mittens','nightgowns','briefs','watches','activewear','jumpsuits rompers','evening dresses','wallets money clips','two piece swimsuits','denim jackets','cropped','trench coats','cardigans','dresses','polo shirts','sweatshirts','robes','demi bras','sports bras','maxi dresses','socks hosiery','push up bras','hair accessories','cocktail dresses','skinny','keychains']
LABEL_MATCH_THRESHOLD = 0.6
# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

FILE = '/Users/saurabhjain/Desktop/cats.csv'
DATA_DIR = None
workbook   = xlsxwriter.Workbook('clothing.xlsx')
sheet = workbook.add_worksheet()
row = 0

category_map = {
'Outerwear': ['clothing|Outerwear', 'clothing|Outerwear|blazers', 'clothing|Outerwear|capes & ponchos',
            'clothing|Outerwear|denim jackets', 'clothing|Outerwear|fur & faux fur', 'clothing|Outerwear|leather & faux leather',
            'clothing|Outerwear|quilted & puffers', 'clothing|Outerwear|raincoats', 'clothing|Outerwear|trench coats', 'clothing|Outerwear|vests'],
'activewear': ['clothing|activewear', 'clothing|activewear|active dresses', 'clothing|activewear|active jackets', 'clothing|activewear|active pants',
            'clothing|activewear|active shirts', 'clothing|activewear|active shorts', 'clothing|activewear|active tops',
            'clothing|activewear|active undergarments'],
'coats & jackets': ['clothing|coats & jackets', 'clothing|coats & jackets|blazers', 'clothing|coats & jackets|leather & fur',
                    'clothing|coats & jackets|quilted & puffers', 'clothing|coats & jackets|raincoats', 'clothing|coats & jackets|trench coats',
                    'clothing|coats & jackets|vests'],
'denim': ['clothing|denim', 'denim|bootcut', 'clothing|denim|boyfriend', 'clothing|denim|cropped', 'clothing|denim|flare',
        'clothing|denim|skinny', 'clothing|denim|straight leg'],
'jumpsuits & rompers': ['clothing|jumpsuits & rompers', 'clothing|jumpsuits, rompers & overalls'],
'pants & shorts': ['clothing|pants & shorts', 'clothing|pants & shorts|casual pants', 'clothing|pants & shorts|dress pants',
                    'clothing|pants & shorts|jeans', 'clothing|pants & shorts|leggings', 'clothing|pants & shorts|shorts'],
'pants' : ['clothing|pants', 'clothing|pants|casual pants', 'clothing|pants|denim', 'clothing|pants|denim|bootcut',
            'clothing|pants|denim|boyfriend', 'clothing|pants|denim|cropped', 'clothing|pants|denim|flare', 'clothing|pants|denim|skinny',
            'clothing|pants|denim|straight', 'clothing|pants|dress pants', 'clothing|pants|leggings'],
'shorts': ['clothing|shorts'],
'skirts': ['clothing|skirts', 'clothing|skirts|maxi skirts', 'clothing|skirts|mid length skirts', 'clothing|skirts|mini skirts'],
'dresses': ['clothing|dresses','clothing|dresses|cocktail dresses', 'clothing|dresses|day dresses', 'clothing|dresses|evening dresses', 'clothing|dresses|maxi dresses'],
'sleepwear & loungewear': ['clothing|sleepwear & loungewear', 'clothing|sleepwear & loungewear|loungewear',
                        'clothing|sleepwear & loungewear|loungewear|loungewear bottoms', 'clothing|sleepwear & loungewear|loungewear|loungewear tops',
                        'clothing|sleepwear & loungewear|nightgowns', 'clothing|sleepwear & loungewear|pajamas', 'clothing|sleepwear & loungewear|robes'],
'suits': ['clothing|suits', 'clothing|suits|pant suits', 'clothing|suits|skirt suits', 'clothing|suits|suit separates', 'clothing|suits|tuxedos'],
'sweaters': ['clothing|sweaters', 'clothing|sweaters|cardigans', 'clothing|sweaters|pullovers', 'clothing|sweaters|turtlenecks'],
'swimwear': ['clothing|swimwear', 'clothing|swimwear|cover-ups', 'clothing|swimwear|one-piece swimsuits', 'clothing|swimwear|swim & board shorts',
            'clothing|swimwear|swim bottoms', 'clothing|swimwear|swimsuit tops', 'clothing|swimwear|two-piece swimsuits'],
'tops': ['clothing|tops', 'clothing|tops|blouses', 'clothing|tops|camisoles & tank tops', 'clothing|tops|dress shirts', 'clothing|tops|polo shirts',
        'clothing|tops|sweatshirts', 'clothing|tops|t-shirts'],
'underwear & intimates': ['clothing|underwear & intimates', 'clothing|underwear & intimates|bras', 'clothing|underwear & intimates|bras|demi bras',
                        'clothing|underwear & intimates|bras|full fit bras', 'clothing|underwear & intimates|bras|push up bras',
                        'clothing|underwear & intimates|bras|sports bras', 'clothing|underwear & intimates|bras|strapless & specialty bras',
                        'clothing|underwear & intimates|garters & belts', 'clothing|underwear & intimates|lingerie',
                        'clothing|underwear & intimates|shapewear', 'clothing|underwear & intimates|slips',
                        'clothing|underwear & intimates|undershirts', 'clothing|underwear & intimates|underwear',
                        'clothing|underwear & intimates|underwear|bikinis', 'clothing|underwear & intimates|underwear|boxer briefs',
                        'clothing|underwear & intimates|underwear|boxers', 'clothing|underwear & intimates|underwear|boyshorts',
                        'clothing|underwe3ar & intimates|underwear|briefs', 'clothing|underwear & intimates|underwear|thongs'],
'uniforms': ['clothing|uniforms', 'clothing|uniforms|professional uniforms'],
#the kids and accessories are needed in the clothing model because currently, we have accessories and kids stuff categoerized as clothing so tensorflow clothing model needs to know these stuff
'kids & baby': ['kids & baby|girls clothing', 'kids & baby|boys clothing'],
'accessories': ['accessories|hair accessories', 'accessories|headwear & kerchiefs', 'accessories|gloves & mittens', 'accessories|belts & buckles', 'accessories|scarves & shawls', 'accessories|hats', 'accessories|sunglasses', 'accessories|neckties', 'accessories|wallets & money clips', 'accessories|socks & hosiery', 'accessories|watches', 'accessories|handbags', 'accessories|jewelry', 'accessories|umbrellas', 'accessories|luggage & bags', 'accessories|keychains', 'accessories|eyeglasses'],
}


table = "suggestions"


def get_best_sr_image_url(image_urls):
    for image in image_urls:
        if image.startswith('180x180|'):
            return image[8:]
    return None


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    # with tf.gfile.FastGFile(os.path.join(
    #         PROJECT_PATH, 'output_graph.pb'), 'rb') as f:
    with tf.gfile.FastGFile(os.path.join(
             PROJECT_PATH, 'data_clothing_l2_classification', 'output', 'output_graph.pb'), 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        del(graph_def.node[1].attr["dct_method"])
        _ = tf.import_graph_def(graph_def, name='')


def _sync_sqlite_file_to_s3(file_path):
    if datetime.now().minute == 0:  # every hour
        # sync the sqlite file to a S3 bucket for Feed Loader to use
        s3_file_suffix = datetime.now().strftime("%Y/%m/%d/%H")
        s3_path = "%s/%s" % (PREDICTION_SQLITE_FILE_S3_LOCATION, s3_file_suffix)
        conn = boto.connect_s3()
        bucket = conn.create_bucket(s3_path)
        with open(file_path, 'r') as f:
            data = f.read()
            key = bucket.new_key("suggestions.db")
            key.set_contents_from_string(data)

def _sanitize_file_name(url):
    return url.split("/")[-1]

def _get_image_as_string(url):
    image = urllib2.urlopen(url, timeout=2)
    # content = requests.get(url)
    file_name = '/tmp/%s.jpg' % _sanitize_file_name(url)
    buffer = StringIO.StringIO()
    buffer.write(image.read())
    im = Image.open(buffer)
    im.save(file_name)
    return file_name


def _get_l2_category(c):
    for l2_cat, cats in category_map.items():
        for cat in cats:
            if c in cat:
                return l2_cat


def _write_to_sheet(sheet, doc_id, name, image, score, human_string, l2s_combined=False):
    print(doc_id, name, image, score, human_string)
    with open(FILE, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([doc_id, name, str(score), human_string, str(l2s_combined)])
    img = _get_image_as_string(image)
    sheet.insert_image("A%s" % row, img)
    sheet.write("B%s" % row, str(doc_id))
    sheet.write("C%s" % row, str(name))
    sheet.write("D%s" % row, str(image))
    sheet.write("E%s" % row, str(score))
    sheet.write("F%s" % row, str(human_string))
    sheet.write("G%s" % row, str(l2s_combined))
    # if row % 20 == 0:
    #     print ("saving...")
    #     import pdb;pdb.set_trace()
    #     sheet.save()


def run_inference_on_images(sess, image, doc_id, name, description, partner_code, row, cursor):
    """Runs inference on an image.

    Args:
      sess: tf session
      image: Image file name.
      doc_id: document id
    Returns:
      Nothing
    """
    try:
        image_data = requests.get(image).content
    except Exception as ex:
        print(ex)
        return
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')  # ADDED

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    feature_set = sess.run(feature_tensor,
                           {'DecodeJpeg/contents:0': image_data})  # ADDED
    feature_set = np.squeeze(feature_set)  # ADDED
    # print(feature_set)  # ADDED
    top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
    # if l3 category is not strong enough, add up scores of L3s to see if combined makes it up to a string L2
    l2_category_scores = defaultdict(int)
    for node_id in top_k:
        # human_string = node_lookup.id_to_string(node_id)
        human_string = LABELS[node_id]
        score = predictions[node_id]
        name = name.replace("'", "")
        if score > LABEL_MATCH_THRESHOLD:
            cursor.execute(
                "INSERT INTO %s VALUES ('%s', '%s', '%s', '%s', '%s', %s, CURRENT_TIMESTAMP)" % (table, doc_id, partner_code, name, image, human_string, score)
            )
            #_write_to_sheet(sheet, doc_id, name, image, score, human_string)
            break
        print('L3: %s, L2: %s' % (human_string, _get_l2_category(human_string)))
        l2_category_scores[_get_l2_category(human_string)]+=score

    if l2_category_scores:
        print(l2_category_scores)
        max_l2_cat, score = sorted(l2_category_scores.items(), key=lambda x:x[1], reverse=True)[0]
        if score > LABEL_MATCH_THRESHOLD:
            cursor.execute(
                "INSERT INTO %s VALUES ('%s', '%s', '%s', '%s', '%s', %s, CURRENT_TIMESTAMP)" % (table, doc_id, partner_code, name, image, human_string, score)
            )
           #_write_to_sheet(sheet, doc_id, name, image, score, max_l2_cat, True)


def extract_features_and_files(image_data, sess):
    pool3 = sess.graph.get_tensor_by_name('incept/pool_3:0')
    features = []
    files = []
    for fname, data in image_data.iteritems():
        try:
            pool3_features = sess.run(pool3, {'incept/DecodeJpeg/contents:0': data})
            features.append(np.squeeze(pool3_features))
            files.append(fname)
        except:
            logging.error("error while processing fname {}".format(fname))
    return features, files


def get_batch(batch):
    """
    Returns: a list of tuples [(docid, imageurl), ....]
    """
    s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
    fq = set()
    fq.add('category:clothing')
    images = []
    q = "*:*"
    results = s.query("category:clothing", fq=fq, fields=['image_url', 'id', 'name', 'description', 'partner_code', 'category'], rows=BATCH_SIZE, start=batch*BATCH_SIZE).results
    image_sets = [(x['image_url'], x['id'], x['name'], x['description'], x['partner_code'], x['category']) for x in results]
    print('products: %s to %s' % ((batch*BATCH_SIZE), (batch+1)*BATCH_SIZE))
    count = 0
    for image_set, doc_id, name, description, partner_code, category in image_sets:
        count += 1
        # has all resolutions. we pick the biggest one for best match (hopefully?)
        best_match_image_url = None
        for image in image_set:
            if image.startswith('180x180|'):
                best_match_image_url = image[8:]
                break
        if not best_match_image_url:
            continue
        images.append((best_match_image_url, doc_id, name, description, partner_code))
    return images


def dress_filter_outs(name, description):
    if (
            ('dress' in name.lower() or 'dress' in description.lower())
            and 'skirt' not in name.lower()
            and 'top' not in name.lower()
    ):
        return True
    return False


def tops_filter_outs(name, description):
    if 'top' in name.lower():
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='Location to the data directory')
    args = parser.parse_args()
    if not args.data_dir:
        raise Exception('Location to data directory is required.')
    DATA_DIR = args.data_dir
    file_path = os.path.join(DATA_DIR, 'refine_clothing.db')
    conn = lite.connect(file_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS %s" % table)
    cur.execute(
        "CREATE TABLE %s(doc_id VARCHAR(50), partner_code VARCHAR(50), name TEXT, image_url TEXT, category VARCHAR(50), confidence DECIMAL, created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP , PRIMARY KEY(doc_id, partner_code))" % table
        )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_partner_code ON %s (partner_code)" % table)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_created_date ON %s (created_date)" % table)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON %s (doc_id)" % table)
    with tf.Session() as sess:
        create_graph()
        batch = 0
        with conn:
            image_tuples = get_batch(batch)
            row = 0
            while image_tuples:
                for image_url, doc_id, name, description, partner_code in image_tuples:
                    try:
                        run_inference_on_images(
                            sess, image_url, doc_id, name, description,
                            partner_code, row, cur
                        )
                        print("committing %s" % doc_id)
                        conn.commit()
                        row+=12
                    except Exception as ex:
                        print(ex)
                        logging.exception("Error running inference on image: %s" % doc_id)
                batch+=1
                image_tuples = get_batch(batch)

if __name__ == '__main__':
    main()
