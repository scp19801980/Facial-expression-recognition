# Facial-expression-recognition
For the facial expression recognition task in complex backgroud, wo proposed a new method based on a multiple branch cross-connected convolutional neural network (MBCC-CNN) for facial expression recognition. The proposed method can fuse the features of each branches more effectively, which solves the problem of insufficient feature extraction of each branches and increases the recognition performance. 

# Files
models--The folder that contains the model. The cnn.py is the proposed model code file.

datasets--Four facial expression datasets. The download address of these datasets are provided in this folder.

utils--Data processing of fer2013 dataset

train.py--Train the model, train_fer2013.py--Train the model about fer2013 dataset.

confusion.py--Data analysis of training and test results

feature_map.py--Generating feature map

heatmap.py--Generating heatmap

# Code execution enviroment
Python 3.6.6

Keras 2.1.4

numpy 1.19.1

matplotlib 3.1.1

sklearn 0.0

opencv-python

tensorflow-gpu 1.5.0

os

itertools


If this project is useful to you, please cite the following paper:
C. Shi, C. Tan and L. Wang, "A facial expression recognition method based on a multibranch cross-connection convolutional neural network," in IEEE Access, doi: 10.1109/ACCESS.2021.3063493.
