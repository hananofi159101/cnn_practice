# -*- coding: utf-8 -*-
"""
fish_multi.py
画像データとメタデータをconcatする
CNNとMLPの多入力モデル

@author: hanano
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
import os
import random
import re
import itertools
import pandas as pd
from keras.layers.core import Activation
from scipy import sparse 
from keras.layers.merge import concatenate


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

X_train_add2 = np.concatenate([X_train_add, X_train], axis=0)

y_train_add2 = np.concatenate([y_train_add, y_train], axis=0)

X_train_add2 = X_train_add2.astype("float32")/255
X_test = X_test.astype("float32")/255




# メタデータ用の処理---------------------------------------------------------------#
#One-hotベクトル→index番号
y_train_add2_meta = np.argmax(y_train_add2,axis=1)
y_test_meta = np.argmax(y_test,axis=1)

X_train_meta = np.zeros((6090, 4))
X_test_meta = np.zeros((254, 4))
fish_data4 = np.loadtxt("fish_data4.csv",       # (900,4)
                  delimiter=",", 
                  skiprows=0,                   # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0,1,2,3)             # 読み込みたい列番号
                 )
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()
fish_data4_norm = mm.fit_transform(fish_data4)

fish_data_0 = fish_data4_norm[0:100]
fish_data_1 = fish_data4_norm[100:200]
fish_data_2 = fish_data4_norm[200:300]
fish_data_3 = fish_data4_norm[300:400]
fish_data_4 = fish_data4_norm[400:500]
fish_data_5 = fish_data4_norm[500:600]
fish_data_6 = fish_data4_norm[600:700]
fish_data_7 = fish_data4_norm[700:800]
fish_data_8 = fish_data4_norm[800:900]

for i in range(6090):
    randomnum = random.randint(0, 99)
    if y_train_add2_meta[i] == 0:
        X_train_meta[i] = fish_data_0[randomnum]
    if y_train_add2_meta[i] == 1:
        X_train_meta[i] = fish_data_1[randomnum]
    if y_train_add2_meta[i] == 2:
        X_train_meta[i] = fish_data_2[randomnum]
    if y_train_add2_meta[i] == 3:
        X_train_meta[i] = fish_data_3[randomnum]
    if y_train_add2_meta[i] == 4:
        X_train_meta[i] = fish_data_4[randomnum]
    if y_train_add2_meta[i] == 5:
        X_train_meta[i] = fish_data_5[randomnum]
    if y_train_add2_meta[i] == 6:
        X_train_meta[i] = fish_data_6[randomnum]
    if y_train_add2_meta[i] == 7:
        X_train_meta[i] = fish_data_7[randomnum]
    if y_train_add2_meta[i] == 8:
        X_train_meta[i] = fish_data_8[randomnum]

for i in range(254):
    randomnum = random.randint(0, 99)
    if y_test_meta[i] == 0:
        X_test_meta[i] = fish_data_0[randomnum]
    if y_test_meta[i] == 1:
        X_test_meta[i] = fish_data_1[randomnum]
    if y_test_meta[i] == 2:
        X_test_meta[i] = fish_data_2[randomnum]
    if y_test_meta[i] == 3:
        X_test_meta[i] = fish_data_3[randomnum]
    if y_test_meta[i] == 4:
        X_test_meta[i] = fish_data_4[randomnum]
    if y_test_meta[i] == 5:
        X_test_meta[i] = fish_data_5[randomnum]
    if y_test_meta[i] == 6:
        X_test_meta[i] = fish_data_6[randomnum]
    if y_test_meta[i] == 7:
        X_test_meta[i] = fish_data_7[randomnum]
    if y_test_meta[i] == 8:
        X_test_meta[i] = fish_data_8[randomnum]
# ----------------------------------------------------------------------------#



def create_mlp(dim, regress=False):
	# define our MLP network
    model = Sequential()
    model.add(Dense(128, input_dim=dim, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    
    return model

mlp = create_mlp(X_train_meta.shape[1], regress=False)
input_tensor = Input(shape=(im_rows, im_cols, im_color)) #224,224,3

# functionalな書き方をする
base_model = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)
x = base_model.output
x = Flatten(input_shape = base_model.output_shape[1:])(x)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu", kernel_initializer="he_normal")(x)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.5)(x)

#MLPとCNNの出力をconcatenateする
combinedInput = concatenate([mlp.output, x])

x = Dense(64, activation="relu")(combinedInput)

x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(9, activation="softmax")(x)



model = Model(inputs=[mlp.input, base_model.input], outputs=x)#新しく学習するとき
for layer in model.layers[:15]: 
    layer.trainable = False
model.summary()


optimizer = optimizers.SGD(lr=0.001, momentum=0.95)


#多クラス分類なら'categorical_crossentropy'
model.compile(loss="categorical_crossentropy",#多クラス分類なので
              optimizer=optimizer,
              metrics=["accuracy"])


early_stopping = EarlyStopping(monitor="acc",min_delta=0.01, patience=15, verbose=1)

hist=model.fit([X_train_meta,X_train_add2],y_train_add2,
              batch_size=32,
              epochs=200,#nb_epochからepochsへ　https://qiita.com/cvusk/items/aa6270301ff2d14fb989
              verbose=1,
              callbacks=[early_stopping],
              validation_split=0.1)


#モデルを評価
score=model.evaluate([X_test_meta,X_test],y_test,verbose=1)
print("正解率=",score[1],"loss=",score[0])


num = 18 #何回目の施行か


fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')
#    axL.grid()
    axL.text(20, 2, "test loss = %ls"%score[0], size = 10)

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="acc for training")
    axR.plot(fit.history['val_acc'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')
#    axR.grid()
    axR.text(20, 0.3, "test acc = %ls"%score[1], size = 10)

plot_history_loss(hist)
plot_history_acc(hist)
fig.savefig('result//lossacc_meta%i.png'%num)
plt.close()



#混同行列を表示
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

preds = model.predict([X_test_meta,X_test]).argmax(axis=1)
cm = confusion_matrix(y_test.argmax(axis=1),preds)
plt.figure(figsize=(6,5))
plot_confusion_matrix(cm, classes=['0','1','2','3','4','5','6','7','8'], title='Confusion matrix, without normalization')
plt.savefig('result//cm_meta%i.png'%num)
plt.figure(figsize=(6,5))
plot_confusion_matrix(cm, classes=['0','1','2','3','4','5','6','7','8'], normalize=True,title='Normalized confusion matrix')
plt.savefig('result//cm_norm_meta%i.png'%num)


#データフレーム型に変換
labels = np.array(['LateolabraxJaponicus','LateolabraxLatus','LateolabraxMaculatus','ThunnusOrientalis','ThunnusAlbacares','ThunnusTonggol','PagrusMajor','EvynnisTumifrons','DentexHypselosomusBleeker'])
cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
cm_labeled.to_csv("result//cm_df_meta%i.csv"%num)



#学習モデルの保存
json_string = model.to_json()
#モデルのファイル名　拡張子.json
open('result//fish_cnn3_meta%i.json'%num, 'w').write(json_string)
#重みファイルの保存 拡張子がhdf5
model.save_weights('result//fish_cnn3_meta%i.hdf5'%num)

