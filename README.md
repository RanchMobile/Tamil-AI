# Tamil AI

This project is created to detect handwritten letters of Tamil Language (South Indian Language) through deep learning nural network (TensorFlow / AI).

Currently this project supports [Inception V3](https://arxiv.org/abs/1512.00567) model (inception_v3).

Sample Traning data is originally from: http://lipitk.sourceforge.net/datasets/tamilchardata.htm

Also used the following document to understand the deep learning: https://web.media.mit.edu/~sra/tamil_cnn.pdf

This project assumes that you already aware of Tensorflow and Python and provided as a starting point.

Let's say we are going to train deep learning AI to learn from 160x160 hand writting, here are the steps to make it work.
We are going to assume you are on Windows Environment for the following steps.

**Prerequisite:**
**Anaconda**
Make sure you have Anaconda is installed for Python 3.6+:
https://www.anaconda.com/download/

**Tensorflow**

Follow the instruction for installing TensorFlow using Anaconda: https://www.tensorflow.org/install/install_windows#installing_with_anaconda

OR 

`C:> conda create -n tensorflow python=3.5` 

`C:> activate tensorflow`

`(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow` 

Test your installation is success by:

`$ python`

run the following commands in python:

>import tensorflow as tf

>hello = tf.constant('Hello, TensorFlow!')

>sess = tf.Session()

>print(sess.run(hello))

Then Exit out of python by entering "quit()" command.

**Step 1: Extract Images**

Unzip the **160x160.zip** file in the **Training Data/Tamil/160x160/** Folder.
Remove all the zip files from that folder.


**Step 2: Training Your Data for image size 160x160:**

`python -m retrain --bottleneck_dir=tf_run/160/bottlenecks --model_dir=tf_run/160/models/ --    summaries_dir=tf_run/160/training_summaries/ --output_graph=tf_run/160/retrained_graph.pb --output_labels=tf_run/160/retrained_labels.txt  --how_many_training_steps=4000 --image_dir=Training Data/Tamil/160x160/`


**Step 3: Validating your own hand written tamil letter by running the following command:**

`python label_image.py --graph=tf_run/160/retrained_graph.pb  --labels=tf_run/160/retrained_labels.txt --dir=Test_Images/160`
