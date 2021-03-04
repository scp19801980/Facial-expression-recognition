#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

input_shape = (48, 48, 1)
num_classes = 7
batch_size = 32
data_path = 'D:/workplace/fighting/database/'
save_path = 'D:/workplace/emotion/feature_map/'
class_names = os.listdir(data_path)
print(class_names)
image_paths = []
for c_name in class_names:
    class_path = data_path + c_name + '/'
    image_name = os.listdir(class_path)
    for i in range(len(image_name)):
        image_name[i] = class_path + image_name[i]
        # image_name[i] = cv2.imread(image_name[i])#
        # image_name[i] = cv2.resize(image_name[i], (48, 48), interpolation=cv2.INTER_CUBIC)#
        # image_name[i] = cv2.cvtColor(image_name[i], cv2.COLOR_BGR2GRAY)#
        image_paths.append(image_name[i])
print(image_paths)
for c_name in class_names:
    save = 'D:/workplace/fighting/' + c_name + '/'
    print(save)
    if not os.path.exists(save):
        os.makedirs(save)

# weights_path = 'D:/workplace/emotion/models/fer2013_mini_XCEPTION.84-0.715.hdf5'
# model = mini_XCEPTION(input_shape, num_classes)
# model.load_weights(weights_path)
model = load_model('D:/workplace/emotion/models/fer2013_mini_XCEPTION.84-0.715.hdf5')
#The Four branches
conv_layer = Model(inputs=model.inputs, outputs=model.get_layer(index=21).output) #77

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    data_path,
    target_size=(input_shape[0], input_shape[1]),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

c = 0
for i in range(len(test_generator)):
    x_test, y_test = test_generator.__getitem__(i)
    conv_output = conv_layer.predict(x_test)
    for j in range(batch_size):
        total_feature_map = conv_output[j, :, :, 0]
        for k in range(1, 144):
            single_feature_maps = conv_output[j,:, :, k]
            total_feature_map = total_feature_map + single_feature_maps
    #
        plt.figure(num=1, figsize=(2, 1.5), dpi=60, clear=True)
        plt.imshow(total_feature_map)

        save_path = 'D:/workplace/fighting/' + image_paths[c][21:-3] + 'png'
        # print(save_path)
        # save_path =  'D:/workplace/emotion/feature_map/' + image_name[i]
        print(save_path)
        plt.savefig(save_path)
        c = c + 1


