# -*- coding: utf-8 -*-
"""
fashionmnist_cnn.py
fashion-mnistをCNNで学習させてみた
CNNの練習

Created on Wed May 22 11:26:33 2019

@author: hanano
"""
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam 
from keras.utils import np_utils
 
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
 
#from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import pandas as pd
 
def build_model():
    model = Sequential()
    
    #畳み込み層の作成
    #1層目の追加　　１０２４個の層を最初に作り、フィルター3*3のフィルターを32個作成
    model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=in_shape)) 
    model.add(Activation("relu"))
    
    #２層目の畳み込み層
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
     #プーリング層
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Dropoutとは過学習を防ぐためのもの　0.25は次のニューロンへのパスをランダムに1/4にするという意味
    model.add(Dropout(0.5))
    
    #３層目の作成
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #４層目の作成
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #プーリング層
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    #５層目
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #6層目
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #プーリング層
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    #平坦化
    model.add(Flatten())
    
    #7　全結合層　FC
    model.add(Dense(100))
    model.add(Activation("relu"))
    
    #Dropout
    model.add(Dropout(0.5))
    
    #8層目　引数nub_classesとは分類の数を定義する。
    model.add(Dense(nub_classes))
    model.add(Activation('softmax'))
    
    #ここまででモデルの層完成
    
    #lossは損失関数を定義するところ
    model.compile(loss="categorical_crossentropy", 
        metrics   = ["accuracy"], 
        optimizer = "adam"
    )
    
    return model
 
def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
 
    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
 
if __name__ == "__main__":
    #　Fashion-MNISTのデータの読み込み
    # 訓練データ６万件、テストデータ１万件
    # 28ピクセル × 28ピクセル
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    in_shape = (28, 28, 1)
    nub_classes = 10
 
    X_train /= 255
    X_test  /= 255
    
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

 
    # 10次元配列に変換　//数字の５ならこんな感じ[0,0,0,0,1,0,0,0,0,0]
    y_train = np_utils.to_categorical(y_train, 10)
    y_test  = np_utils.to_categorical(y_test, 10)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
 
    # データで訓練
    model = build_model()
    history = model.fit(X_train, y_train, 
        nb_epoch=50, #学習させる回数　回数はお好みで　pytyonのnb_epochとはrangeの繰り返しのこと
        batch_size=128, #無作為に１２８画像取得している。数字はなんでも良い
        validation_data=(X_test, y_test)
    )
 
    #学習モデルの保存
    json_string = model.to_json()
    #モデルのファイル名　拡張子.json
    open('mnist.json', 'w').write(json_string)
    #重みファイルの保存 拡張子がhdf5
    model.save_weights('mnist.hdf5')
 
    # モデルの評価を行う
    score = model.evaluate(X_test, y_test, verbose=1)
 
    print('loss=', score[0])
    print('accuracy=', score[1])
    
    # modelに学習させた時の変化の様子をplot
    plot_history(history)
