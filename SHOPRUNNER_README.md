# https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html

# build the binary for retraining
bazel build tensorflow/examples/image_retraining:retrain


#To train:
sudo bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir $HOME/data_electronics/train --bottleneck_dir $HOME/data_electronics/output/bottleneck  --model_dir $HOME/data_electronics/output  --learning_rate 0.01 --how_many_training_steps 100000 $HOME/retrain_logs

#To label image
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=data/labels_file.txt --output_layer=final_result --image=/Users/saurabhjain/Desktop/shoe.jpg

#To label and write to sqlite file used for other ShopRunner systems
python scripts/sr_classify_image.py --data-dir=/Users/saurabhjain/tensorflow/data_electronics --sqlite-file=/Users/saurabhjain/tensorflow/data_electronics/suggested_tv.db --label-to-find=tv --solr-query=name_search:TV --threshold=0.95


# classify TVs
python scripts/sr_classify_image.py  --data-dir=/home/ubuntu/tensorflow/tensorflow/data_electronics --label-file=/home/ubuntu/tensorflow/tensorflow/data_electronics/output/output_labels.txt --solr-query=name_search:TV --threshold=0.95 --label-to-find=tv --sqlite-file=suggested_tvs.db --sync-s3=y

# classify Shoes
python scripts/sr_classify_image.py  --data-dir=/home/ubuntu/tensorflow/tensorflow/data_shoes \
--label-file=/home/ubuntu/tensorflow/tensorflow/data_shoes/output/output_labels.txt \
--solr-query="name_search:shoe OR name_search:boot OR name_search:sandal OR \
    name_search:flat OR name_search:pump OR name_search:oxford OR name_search:flip \
    OR name_search:thong OR name_search:slipper OR name_search:loafer" \
--threshold=0.95
--sqlite-file=suggested_shoes.db --sync-s3=y