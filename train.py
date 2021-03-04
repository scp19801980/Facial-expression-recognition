#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from vgg16 import mini_XCEPTION
from cnn_generator import CaptchaSequence
from keras.models import *

# Input model image shape
img_width, img_height = 48, 48

train_data_dir = 'D:/workplace/emotion/datasets/fer2013plus/train'  #Database training set path
validation_data_dir = 'D:/workplace/emotion/datasets/fer2013plus/valid'#Verification set path
test_dir = 'D:/workplace/emotion/datasets/fer2013plus/test'# Test set path
train_samples = 25045
validation_samples = 3191
epochs = 1000
batch_size = 32
input_shape = (48, 48, 3)
num_calsses = 8
patience = 30
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_`height)
# else:
#     input_shape = (img_width, img_height, 3)
#Loading model
model = mini_XCEPTION(input_shape, num_calsses)
model.summary()

from keras import optimizers
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
# sgd = optimizers.SGD()
model.compile(loss='categorical_crossentropy',  # Multiclassification
              optimizer='sgd',
              metrics=['accuracy'])
#Data enhancement
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,  # The maximum value of random rotation is 10
    width_shift_range=0.1,  # Scale range of horizontal translation
    height_shift_range=0.1,  # Scale range of vertical translation
    shear_range=0.1, #Horizontal random flip input
    zoom_range=0.1, # Random scaling range
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # Multiclassification

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

labels = train_generator.class_indices # Get the label corresponding to the classification
print(labels)

filename = 'D:\\workplace\\emotion\\models\\FER+_vgg_3\\model_train_new.csv'
base_path = 'D:\\workplace\\emotion\\models\\FER+_vgg_3\\'
log_file_path = base_path + "fer2013plus" + '_emotion_training.log'  # Loading path
csv_logger = CSVLogger(filename, append=False)  # Overlay the existing file (append = false) to stream the epoch result to the callback of the CSV file.
early_stop = EarlyStopping('val_loss', patience=patience)  # When the early stop is activated (if loss is not decreased compared with the previous epoch training), the training will be stopped after 100 epochs
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,  # When the index stopped improving, the learning rate decreased by 0.1 every time
                                  patience=int(patience / 4), verbose=1)  # Output progress bar
trained_models_path = base_path + "fer2013plus" + '_model'  # Model path
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'  # The name of the model

model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,
                                       save_best_only=True)  # When set to true, only the best performing models on the validation set are saved
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    verbose=2,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size)
