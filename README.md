# CV_Flowers
# Image Classification with ResNet-50 on the Oxford Flowers Dataset

This project is a TensorFlow implementation for image classification using the **ResNet-50** model, fine-tuned on the **Oxford 102 Flowers** dataset. The dataset is loaded from `.mat` files, and the model is trained to classify images of flowers into 102 different categories.

## Project Overview

This project trains a **ResNet-50** deep learning model to classify flower images into 102 categories. The model is trained using the **Oxford 102 Flowers** dataset, which consists of `.jpg` images and `.mat` files for labels and dataset splits (training, validation, and test sets). The goal is to leverage transfer learning by fine-tuning the ResNet-50 model, pre-trained on ImageNet.

## Requirements

Make sure you have the following dependencies installed:

1. **Python 3.x**
2. **TensorFlow 2.x**
3. **SciPy** (for loading `.mat` files)
4. **Oxford 102 Flowers Dataset**: Download the `.mat` and `.jpg` files from [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

## Dataset

The dataset consists of `.jpg` images and the following `.mat` files:

1. **`imagelabels.mat`**: Contains the labels for each image.
2. **`setid.mat`**: Contains the split information for training, validation, and test sets.

The images are stored in the `jpg/` folder, with file names in the format: `image_XXXXX.jpg`.

## Model Architecture

The model architecture is based on **ResNet-50** with the following modifications:

- The model is pre-trained on **ImageNet**.
- The top layer of ResNet-50 is removed, and a **Global Average Pooling** layer is added.
- A **Dense** layer with 102 units (for 102 classes) and a **softmax** activation function is added.


