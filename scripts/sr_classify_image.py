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
import re
import requests
import csv
import sys
import tarfile
import boto.dynamodb
import boto
import sqlite3 as lite
import numpy as np
import logging
from six.moves import urllib
import tensorflow as tf
import argparse
from datetime import datetime
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 1000
COMMIT_BATCH_SIZE = 100
MODULE_PATH = os.path.abspath(os.path.split(__file__)[0])
PROJECT_PATH = "/".join(MODULE_PATH.split("/")[:-1])
NUM_TOP_PREDICTIONS = 5
PREDICTION_SQLITE_FILE_S3_LOCATION = 'net.shoprunner.prd.jarvis/tensorflow_recommendations'
LABELS = ["bags","womens intimates","shoes","mens suits","tops","mens shirt","womens outerwear","mens underwears","mens outerwear","bottoms","womens swimwear","dresses"]
LABELS_ELECTRONICS = ['speakers','laptop computers','cell phones','headphones','tv','tablets','desktop computers','monitors','mp3 player']
LABEL_MATCH_THRESHOLD = 0.5
# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

DATA_DIR = None

def get_best_sr_image_url(image_urls):
    for image in image_urls:
        if image.startswith('180x180|'):
            return image[8:]
    return None


def create_graph(data_dir):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    # with tf.gfile.FastGFile(os.path.join(
    #         PROJECT_PATH, 'output_graph.pb'), 'rb') as f:
    with tf.gfile.FastGFile(os.path.join(
             data_dir, 'output', 'output_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        del(graph_def.node[1].attr["dct_method"])
        _ = tf.import_graph_def(graph_def, name='')


def _get_sqlite_db_name_from_file_path(file_path):
    return file_path.split('/')[-1]


def _sync_sqlite_file_to_s3(file_path):
    if datetime.now().minute == 0:  # every hour
        # sync the sqlite file to a S3 bucket for Feed Loader to use
        s3_file_suffix = datetime.now().strftime("%Y/%m/%d/%H")
        s3_path = "%s/%s" % (PREDICTION_SQLITE_FILE_S3_LOCATION, s3_file_suffix)
        conn = boto.connect_s3()
        bucket = conn.create_bucket(s3_path)
        with open(file_path, 'r') as f:
            data = f.read()
            key = bucket.new_key(_get_sqlite_db_name_from_file_path(file_path))
            key.set_contents_from_string(data)


def run_inference_on_images(sess, image, doc_id, name, description, partner_code,
                            table=None, cursor=None, file_path=None, label_to_find="dresses", threshold=LABEL_MATCH_THRESHOLD):
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

    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    feature_set = sess.run(feature_tensor,
                           {'DecodeJpeg/contents:0': image_data})  # ADDED
    feature_set = np.squeeze(feature_set)  # ADDED
    # print(feature_set)  # ADDED
    top_k = predictions.argsort()[-NUM_TOP_PREDICTIONS:][::-1]
    for node_id in top_k:
        human_string = LABELS_ELECTRONICS[node_id]
        score = predictions[node_id]
        filter_output = True
        if label_to_find == 'dresses':
            filter_output = filter_output and dress_filter_outs(name, description)

        if (
                human_string == label_to_find
                and filter_output
        ):
            if score > threshold:
                print(doc_id, name, image, score)

                cursor.execute(
                    "INSERT INTO %s VALUES ('%s', '%s', '%s', %s, CURRENT_TIMESTAMP)" % (table, doc_id, partner_code, label_to_find, score)
                )


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


def get_batch(batch, solr_category):
    """
    Returns: a list of tuples [(docid, imageurl), ....]
    """
    s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
    fq = set()
    images = []

    q = "category_all:%s" % solr_category
    results = s.query(q, fq=fq, fields=['image_url', 'id', 'name', 'description', 'partner_code'], rows=BATCH_SIZE, start=batch*BATCH_SIZE).results
    image_sets = [(x['image_url'], x['id'], x['name'], x['description'], x['partner_code']) for x in results]
    print('products: %s to %s' % ((batch*BATCH_SIZE), (batch+1)*BATCH_SIZE))
    count = 0
    for image_set, doc_id, name, description, partner_code in image_sets:
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
    parser.add_argument('--sqlite-file', help='Sqlite file location where the results of categorization are written to and used later')
    parser.add_argument('--table-name', help='Sqlite table name')
    parser.add_argument('--label-to-find', help='Label to find')
    parser.add_argument('--solr-category', help='SOLR category to look for the corpus and match the images')
    parser.add_argument('--threshold', help='Minimum threshold to match a label')

    args = parser.parse_args()

    if not args.data_dir:
        raise Exception('Location to data directory is required.')

    DATA_DIR = args.data_dir

    with tf.Session() as sess:
        create_graph(DATA_DIR)

        table =  args.table_name or "ProductCategory"
        file_path = args.sqlite_file or 'suggested_dresses.db'
        label_to_find = args.label_to_find or 'dresses'
        solr_category = args.solr_category or 'clothing'
        threshold = args.threshold or LABEL_MATCH_THRESHOLD
        conn = lite.connect(file_path)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS %s" % table)
        cur.execute(
            """
                CREATE TABLE %s(doc_id VARCHAR(50), partner_code VARCHAR(50),
                category VARCHAR(50), confidence DECIMAL, created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(doc_id, partner_code))""" % table
            )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_partner_code ON %s (partner_code)" % table)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_created_date ON %s (partner_code)" % table)
        batch = 0
        #with conn:
        image_tuples = get_batch(batch, solr_category)
        while image_tuples:
            for image_url, doc_id, name, description, partner_code in image_tuples:
                try:
                    run_inference_on_images(
                        sess, image_url, doc_id, name, description,
                        partner_code, table=table,cursor=cur, file_path=file_path,
                        label_to_find=label_to_find, threshold=threshold
                    )

                    conn.commit()
                except Exception as ex:
                    logging.exception("Error running inference on image: %s" % doc_id)
            batch+=1
            image_tuples = get_batch(batch, solr_category)

            # sync the
            _sync_sqlite_file_to_s3(file_path)

if __name__ == '__main__':
    main()
