#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__: "TC"
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

input_shape = (48, 48, 1)
validation_split = 0.1
model = load_model('D:\\workplace\\fighting\\trained_models\\71.5\\fer2013_mini_XCEPTION.84-0.715.hdf5')
datasets = ['fer2013']
# for循环实现回调，加载数据集
for dataset_name in datasets:
    print('Training dataset:', dataset_name)  # 遍历循环，在数据dataset中逐个取值存入变量dataset_name中，然后运行循环体

    data_loader = DataManager(dataset_name, image_size=input_shape[:2])  # 自定义DataManager函数实现根据数据集name进行加载，dataloder返回一个迭代器  对输入张量进行切片操作
    faces, emotions = data_loader.get_data()  # 自定义get_data函数根据不同数据集name得到各自的ground truth data
    #faces = cv2.resize(faces.astype('uint8'), input_shape, cv2.COLOR_GRAY2BGR) ###!!!!
    faces = preprocess_input(faces)  # 自定义preprocess_input函数：处理输入的数据，先转为float32类型然后/ 255.0
    num_samples, num_classes = emotions.shape  # shape函数读取矩阵的长度
    train_data, val_data = split_data(faces, emotions, validation_split)  # 自定义split_data对数据整理各取所得train_data、 val_data
    train_faces, train_emotions = train_data
    val_x, val_y = val_data
    score = model.evaluate(val_x, val_y, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    test_true = np.argmax(val_y, axis=1)
    test_pred = np.argmax(model.predict(val_x), axis=1)
    print(test_pred)
    print(classification_report(test_true, test_pred))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='真实类别',
           xlabel='预测类别')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
plt.rc('font', family='SimHei', size=13)
# #根据预测结果显示对应的文字label
classes = ['Angry',  'Disgust', 'Fear', 'Happy',  'Sad',  'Surprise',  'Neutral']

plot_confusion_matrix(test_true, test_pred, classes=classes, normalize=True, title='混淆矩阵')
plt.show()


classes = [x for x in range(7)]
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