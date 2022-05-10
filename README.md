# Face Mask Detector

## Introduction

This project uses a Deep Neural Network, more specifically a
**Convolutional Neural Network(CNN)**, to differentiate between images
of people *with* and *without masks*.

The CNN manages to get an accuracy of **98.2% on the training set**
and **97.3% on the test set**.

Then the stored weights of this CNN are used to classify as *mask* or *no mask*,
in real time, using OpenCV. With the webcam capturing the video,
the frames are preprocessed and fed to the model to accomplish this task.
The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction

The model is capable of predicting multiple faces with or without masks at the same time.

## Dataset

The dataset is a collection of images of people with and without masks.

The data can be found directly in this repository. (folder 'Data')
or also can be downloaded from [here](https://data-flair.training/blogs/download-face-mask-data/)

There are 1314 training images and 194 test images divided into two catgories, with and without mask.
