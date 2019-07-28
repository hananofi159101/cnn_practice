# -*- coding: utf-8 -*-
"""
vgg16_to_rf.py
VGG16を特徴量として使ってみる
抽出した特徴量をrfで学習させる

Created on Tue Jun 25 21:57:48 2019

@author: eupho
"""


from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.utils import np_utils
from keras import backend as K
import os
import seaborn as sn
import re
import itertools
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


#サイズを指定
im_rows = 224
im_cols = 224
im_color = 3
in_shape = (im_rows,im_cols,im_color)
nb_classes = 9


#ImageDataGenerator と　画像を水増しする関数---------------------------------------#
datagen = ImageDataGenerator(
                            rotation_range=20,
                            width_shift_range=0.01,
                            height_shift_range=0.01,
                            brightness_range=(0.5, 1.0),
                            zoom_range=0.2,#拡大縮小
                            shear_range=5,#せん断(引き伸ばし)
                            channel_shift_range=5,
                            horizontal_flip = True,
                            vertical_flip = True
                            )

def images_gen(x_list,y_list):
    output_dir = 'test'
    x_list_add = []
    y_list_add = []
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    for x ,y in zip(x_list,y_list):#xは(3, width, height)で受け取る
        x = x.reshape((1,) + x.shape)  #(1, 3, width, height)に変換する
        
        i = 0
        for batch in datagen.flow(x, batch_size=32, save_to_dir=output_dir, save_prefix='img', save_format='jpg'):
            batch = batch.astype(np.uint8)#データ型を揃える
            batch = batch.reshape((224, 224, 3))
            x_list_add.append(batch)
            y_list_add.append(y)
            i += 1
            if i > 4:#１枚から5枚作る
                break             
    x_np_add = np.array(x_list_add)
    y_np_add = np.array(y_list_add)
    

            
    return x_np_add,y_np_add
#------------------------------------------------------------------------------#




#写真データを読み込みの関数--------------------------------------------------------#
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
#------------------------------------------------------------------------------#
    
    

X = []
Y = []
count = 0

filelist = ["LateolabraxJaponicus_resize_224","LateolabraxLatus_resize_224","LateolabraxMaculatus_resize_224","ThunnusOrientalis_resize_224","ThunnusAlbacares_resize_224","ThunnusTonggol_resize_224","PagrusMajor_resize_224","EvynnisTumifrons_resize_224","DentexHypselosomusBleeker_resize_224"]
for dir in filelist:
    for picture in list_pictures(dir + '/'):
        img = img_to_array(load_img(picture, target_size=(im_rows,im_cols)))
        X.append(img)
        Y.append(count)
    count += 1


# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)

# 画素値を0から1の範囲に変換
#X = X.astype('float32')
#X = X / 255.0 #最大値で割ることでデータを正規化する

# クラスの形式を変換
Y = np_utils.to_categorical(Y, 9)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=111)
#inputtensorをかく


#trainデータを水増しする
X_train_add,y_train_add = images_gen(X_train,y_train)

#水増しなし用
#X_train = X_train.astype("float32")/255
#X_test = X_test.astype("float32")/255


X_train_add2 = np.concatenate([X_train_add, X_train], axis=0)

y_train_add2 = np.concatenate([y_train_add, y_train], axis=0)

X_train_add2 = X_train_add2.astype("float32")/255
X_test = X_test.astype("float32")/255

#ワンホット表現　→　index番号(カテゴリ番号)
y_train_add2 = np.argmax(y_train_add2,axis=1)
y_test = np.argmax(y_test,axis=1)



print(X_train.shape)    
print(y_train.shape)
print(X_test.shape) 
print(y_test.shape)
print(X_train_add.shape)
print(y_train_add.shape)
print(X_train_add2.shape)
print(y_train_add2.shape)







# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=(im_rows, im_cols, 3))
model_vgg16 = VGG16(weights='imagenet', include_top=False, pooling="avg",input_tensor=input_tensor)
X_train_vgg16 = model_vgg16.predict(X_train_add2)
X_test_vgg16 = model_vgg16.predict(X_test)
#特徴量を抽出



rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False).fit(X_train_vgg16,y_train_add2)
#print("acc = {}".format(accuracy_score(rf.predict(X_train_vgg16), y_train_add2)))
print("acc = {}".format(accuracy_score(rf.predict(X_test_vgg16), y_test)))


#混同行列を表示
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.figure(figsize=(10,10))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


preds = rf.predict(X_test_vgg16)
cm = confusion_matrix(y_test,preds)
plt.figure(figsize=(6,5))
plot_confusion_matrix(cm, classes=['0','1','2','3','4','5','6','7','8'],title='Confusion matrix, without normalization')
plt.savefig('result//cm_rf.png')
plt.figure(figsize=(6,5))
plot_confusion_matrix(cm, classes=['0','1','2','3','4','5','6','7','8'], normalize=True,title='Normalized confusion matrix')
plt.savefig('result//cm_norm_rf.png')



