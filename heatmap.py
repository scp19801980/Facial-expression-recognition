#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
import numpy as np
import cv2, os
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from keras.models import load_model

K.clear_session()
input_shape = (48,48, 3)
num_classes = 8
img_path = 'D:/workplace/emotion/fer.png'
model = load_model('D:\\workplace\\emotion\\models\\f\\fer2013plus_model.83-0.8810.hdf5')
path = 'D:/workplace/emotion/f.jpg'
# model = mini_XCEPTION(input_shape=input_shape, num_classes=num_classes)
# model.load_weights(weight_path)

rescale = 1. / 255

img = cv2.imread(img_path)
img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
# img = image.load_img(img_path, target_size=(48, 48))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = load_img(img_path, target_size=(48, 48))
x = img_to_array(img)
x *= rescale
x = np.expand_dims(x, axis=0)

preds = model.predict(x)

index = np.argmax(preds[0])
print(index)
output = model.output[:, index]
last_conv_layer = model.get_layer(index=-40)
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(40):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 1 + img

cv2.imwrite(path, superimposed_img)
