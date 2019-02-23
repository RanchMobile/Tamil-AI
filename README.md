# Tamil AI / Tamil Artificial Intelligence

This project is created to detect handwritten letters of Tamil Language (South Indian Language) through deep learning nural network (TensorFlow / AI).

Currently this project supports [Inception V3](https://arxiv.org/abs/1512.00567) model (inception_v3).

Sample Traning data is originally from:
> http://shiftleft.com/mirrors/www.hpl.hp.com/india/research/penhw-resources/tamil-iwfhr06-train.html

You can download the original images:
> http://shiftleft.com/mirrors/www.hpl.hp.com/india/research/penhw-resources/hpl-tamil-iwfhr06-train-offline.tar.gz

some of the images are resized for training in 160x160, 64x64, 32x32 size in JPG format. 

**NOTE: not all the images are currently converted.**

Also used the following document to understand the deep learning: https://web.media.mit.edu/~sra/tamil_cnn.pdf

This project assumes that you already aware of Tensorflow and Python and provided as a starting point.

Let's say we are going to train deep learning AI to learn from 160x160 hand writting, here are the steps to make it work.
We are going to assume you are on Windows Environment for the following steps.

## Prerequisite

### Anaconda

Make sure you have Anaconda is installed for Python 3.5+:
https://www.anaconda.com/download/

### Create and Activate Tensorflow in Anaconda

Follow the instruction for installing TensorFlow using Anaconda: https://www.tensorflow.org/install/install_windows#installing_with_anaconda

OR 

`C:> conda create -n tensorflow python=3.5` 

`C:> activate tensorflow`

`(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow` 

### Validate Installation
Test your installation by runing the following commands using the test.py file (provided) in python:

`(tensorflow)C:> python test.py`

## Steps
### Step 1: Clone this Project
Clone this Project to a local folder and go to the folder.

### Step 2: Extract Images
Unzip the **160x160.zip** file in the **Training Data/Tamil/160x160/** Folder.
Remove all the zip files from that folder.
You can also use [Tamil Font to Handwriting Image](https://github.com/RanchMobile/Tamil-Font-to-Image-AI) project to generate more images.

### Step 3: Activate TensorFlow
Go to command Prompt **(Start->Run->CMD)**
then Type the following command:

`C:> activate tensorflow`

### Step 4: Training Your Data for image size 160x160
Start training your AI by using the following command:

`(tensorflow)C:> python -m retrain --how_many_training_steps=6000`

larger the steps for training using "how_many_training_steps" is better accuracy of the results going to be. 

### Step 5: Validating your own hand written tamil letter by running the following command
Validate by testing by providing your own 160 x 160 image. or use the sample from **Test_Images/160** Folder.

`(tensorflow)C:> python label_image.py --test_data_dir=Test_Images/160`
