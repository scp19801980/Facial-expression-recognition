#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import itertools
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model_file = 'D:/workplace/emotion/models/ff_2/fer2013plus_model.83-0.8810.hdf5'
test_dir = 'D:/workplace/emotion/datasets/fer2013plus/valid/'

input_shape = (48, 48, 3)
model = load_model(model_file)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=3137,
    class_mode='categorical')

#test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
#print('test acc: %.3f%%' % test_acc)
labels = (test_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

x_test, y_test = test_generator.__getitem__(0)
test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(model.predict(x_test), axis=1)
print(test_pred)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))#输出模型测试平均值

preds = model.predict(x_test)
print(classification_report(test_true, test_pred)) # Output classification report (accuracy rate, regression rate, F1)_ score）

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

# compute confusion matrix
cnf_matrix = confusion_matrix(test_true, test_pred)
np.set_printoptions(precision=45)
# plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=labels, title='Normalized confusion matrix')
plt.show()

classes = [x for x in range(9)]
confusion = confusion_matrix(test_true, test_pred, classes)
print(confusion)

list_diag = np.diag(confusion)
print(list_diag)

list_raw_sum = np.sum(confusion, axis=1)
print(list_raw_sum)

each_acc = np.nan_to_num(list_diag.astype('Float32')/list_raw_sum.astype('Float32'))
print(each_acc)

ave_acc = np.mean(each_acc)
print(ave_acc)