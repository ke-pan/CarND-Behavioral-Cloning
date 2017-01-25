# Behavioral Cloning

This is one of Udacity self-driven car nanodegree projects. Purpose of this project is to train a neural network to clone driving behavior.

## Data Augmentation

Data generated by simulator is limited and bias. Most of the data have small angle. This make the predictions tend to be zero.

Inspired by [Mohan Karthik](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.cnd8p0v7s), [Nick Hortovanyi](https://medium.com/@NickHortovanyi/clone-driving-behaviour-17a809cd400b#.amidckol1) and [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.vmxld9pwn), I used several ways to augment the data.

1. Using left, right camera images.
2. Making a horizontal transform to original images.
3. Mirroring the original images.
4. Modifying brightness of original images.

## Crop and Resize

I crop the original images to remove useless part. Then I resize images to 64x64.

## Architecture

It's [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) architecture.

layer 1: Normalization layer

layer 2: Convolution layer with 16 8x8 filters, 4x4 strides, same padding, elu activation.

layer 3: Convolution layer with 32 5x5 filters, 2x2 strides, same padding, elu activation.

layer 3: Convolution layer with 64 5x5 filters, 2x2 strides, same padding, elu activation.

layer 4: Flatten layer with dropout, elu activation.

layer 5: 512 outputs fully connected layer with dropout, elu activation.

layer 6: 1 output fully connected layer. This is output of steering.

## Training

Mean Square Error is used as loss function. 100 epochs with 5000 data generated per epoch.

Because through data augmentation ways mentioned above, the training data is actually infinite. I don't use validate dataset.

## What I learn from this project

The most import thing I learn from this project is the data augmentation. Before I augment the images, I just passed first corner of track 1.
