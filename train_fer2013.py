#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from Alexnet import mini_XCEPTION
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input
batch_size = 32
num_epochs = 1000
input_shape = (48, 48, 1)
validation_split = 0.1
verbose = 1
num_classes = 7
patience = 30
base_path = 'D:/workplace/emotion/models/a_mbcc/'

# Data generator calls imagedatagenerator function to enhance real-time data and generate small batch of image data.
data_generator = ImageDataGenerator(
    featurewise_center=False,  #  Set input mean to 0 over the dataset, feature-wise.
    featurewise_std_normalization=False,  # by std of the dataset, feature-wise.
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
   # rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest')

# Model parameters / compilation
model = mini_XCEPTION(input_shape, num_classes)
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()  # Output the parameters of each layer of the model

# Training fer2013 dataset
datasets = ['fer2013']
# For loop to implement callback and load data set
for dataset_name in datasets:
    print('Training dataset:', dataset_name)  # Traverse the loop and store the values one by one in the data dataset into the variable dataset_ Name, and then run the loop body
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=(patience / 4), verbose=1)

    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.3f}.hdf5'

    model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
    # Loading data sets
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    #faces = cv2.resize(faces.astype('uint8'), input_shape, cv2.COLOR_GRAY2BGR) ###!!!!
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    print(num_samples)
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    # Calling fit from training model_ Training model of generator function
    history = model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                                      batch_size),
                                steps_per_epoch=len(train_faces) / batch_size,
                                epochs=num_epochs, verbose=2, callbacks=callbacks,
                                validation_data=val_data)
