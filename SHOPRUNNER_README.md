# https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html

To train:
sudo bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /Users/saurabhjain/tensorflow/data/train --bottleneck_dir /Users/saurabhjain/tensorflow/data/output/bottleneck  --model_dir /Users/saurabhjain/tensorflow/data/output  --learning_rate 0.01

To label image
bazel-bin/tensorflow/examples/label_image/label_image --graph=/tmp/output_graph.pb --labels=data/labels_file.txt --output_layer=final_result --image=/Users/saurabhjain/Desktop/shoe.jpg
or
python sr_classify_image.pyyy
