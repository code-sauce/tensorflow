# https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html

# build the binary for retraining
bazel build -c opt --copt=-mavx tensorflow/examples/image_retraining:retrain


#To train:
sudo bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /Users/saurabhjain/tensorflow/data_electronics/train --bottleneck_dir /Users/saurabhjain/tensorflow/data_electronics/output/bottleneck  --model_dir /Users/saurabhjain/tensorflow/data_electronics/output  --learning_rate 0.01 --how_many_training_steps 100000

#To label image
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=data/labels_file.txt --output_layer=final_result --image=/Users/saurabhjain/Desktop/shoe.jpg

#To label and write to sqlite file used for other ShopRunner systems
python scripts/sr_classify_image.py --data-dir=/Users/saurabhjain/tensorflow/data_electronics --sqlite-file=/Users/saurabhjain/tensorflow/data_electronics/suggested_tv.db --label-to-find=tv --solr-query=name_search:electronics --threshold=0.8
