#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
from keras.models import *
from keras.layers import *
from keras import regularizers
#     71.5(le-2)
def one1(x, params):
    residul = Conv2D(params[0], (3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    residul = BatchNormalization()(residul)

    x = Conv2D(params[1], (3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(params[2], (3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = add([residul, x])
    x = Activation('relu')(x)
    # x = add([residul, x])
    return x
#
def one2(x, params):
    x1 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x2)
    x3 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x1)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x3)
    y1 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x2)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y2 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y3 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x3)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y4 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x3)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)
    x = add([y1, y2, y3, y4])
    x = Conv2D(params,(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    return x

def one3(x, params):

    x = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = Conv2D(params, (1, 1), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x3 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x4 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(params, (3, 3), strides=(1, 1), padding='same',
                kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x = add([x1, x2, x3, x4])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    return x
def mini_XCEPTION(input_shape, num_classes=7):
    input = Input(shape=(48, 48,1))

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = one1(x, [72, 72, 72])
    x = one3(x, 72)
    x = one2(x, 72)
    x = one1(x, [144, 144, 144])
    x = one3(x, 144)
    x = one2(x, 144)
    x = Conv2D(288, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(7, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model

if __name__ == '__main__':
    input_shape = (48, 48, 1)
    num_classes = 7
    model = mini_XCEPTION(input_shape, num_classes)
    model.summary()