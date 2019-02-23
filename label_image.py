# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb  --labels=retrained_labels.txt   --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys
import time

import tensorflow as tf

parser = argparse.ArgumentParser()
#parser.add_argument(
#    '--image', required=True, type=str, help='Absolute path to image file.')
parser.add_argument(
    '--test_data_dir', 
    type=str, 
    default="Test_Images/160",
    help='Absolute path to images folder.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    type=str,
    default="tf_run/160/retrained_graph.pb",
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    type=str,
    default="tf_run/160/retrained_labels.txt",
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(dir, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    for subdir, dirs, files in os.walk(FLAGS.test_data_dir):
												for file in files:
																	start_time = time.time()
																	image_path = os.path.join(subdir, file)
																	image_data = load_image(image_path)  
																	predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})
																	top_k = predictions.argsort()[-num_top_predictions:][::-1]	
																	print(image_path)
																	
																	i=0
																	human_string = labels[top_k[i]]
																	score = predictions[top_k[i]]*100																	
																	print('%s [%.2f%%]' % ( human_string, score))
																	
																	i+=1
																	human_string = labels[top_k[i]]
																	score = predictions[top_k[i]]*100
																	print('%s [%.2f%%]' % ( human_string, score))
																	
																	i+=1
																	human_string = labels[top_k[i]]
																	score = predictions[top_k[i]]*100
																	print('%s [%.2f%%]' % ( human_string, score))
																	print("--- prediction in %s second(s) ---" % (time.time() - start_time))
																	print('')
    #for node_id in top_k:
    #  human_string = labels[node_id]
    #  score = predictions[node_id]
    #  print('%s (score = %.5f)' % (human_string, score))	

    return 0
		
def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  #if not tf.gfile.Exists(FLAGS.image):
  #  tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  
  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)
  
  if tf.gfile.Exists(FLAGS.test_data_dir):
								run_graph(FLAGS.test_data_dir, labels, FLAGS.input_layer, FLAGS.output_layer,FLAGS.num_top_predictions)								
#																								
		
		
  # load image
  #image_data = load_image(FLAGS.image)
  
  #run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
  #          FLAGS.num_top_predictions)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
