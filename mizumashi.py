# -*- coding: utf-8 -*-
"""
mizumashi.py 画像水増し用プログラム
Created on Wed Jun 12 14:53:42 2019

@author: hanano
"""

#練習


from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
import math

filelist = ["LateolabraxJaponicus_resize_resize2","LateolabraxLatus_resize_resize2","LateolabraxMaculatus_resize_resize2","ThunnusOrientalis_resize_resize2","ThunnusAlbacares_resize_resize2","ThunnusTonggol_resize_resize2","PagrusMajor_resize_resize2","EvynnisTumifrons_resize_resize2","DentexHypselosomusBleeker_resize_resize2"]
#filelist = ["LateolabraxMaculatus_resize_resize2","ThunnusOrientalis_resize_resize2","ThunnusAlbacares_resize_resize2","ThunnusTonggol_resize_resize2","PagrusMajor_resize_resize2","EvynnisTumifrons_resize_resize2","DentexHypselosomusBleeker_resize_resize2"]
#filelist = ["LateolabraxJaponicus2"]
for dir in filelist:
    
    files = os.listdir(dir)
    
    count = 0       #count初期化
    tmpnum = 0      #tmpnum初期化
    NOS = 0         #NOS初期化
    
    for file in files:
        index = re.search('.',file) #'.'が含まれるファイルを数える（'.jpg'と）したらjpgファイルだけ数える
        if index:
            count = count + 1
            
    print(count)
    tmpnum = 500 / count 
    NOS = math.ceil(tmpnum) #NOS =　Number of sheets
    print(NOS)

    
    
    
    # 入力ディレクトリを作成
    input_dir = dir
    files = glob.glob(input_dir + '\*')
    print(files)
    
    # 出力ディレクトリを作成
    output_dir = dir + '_mizumashi'
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    
    
    for i, file in enumerate(files):
            img = load_img(file)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
        
            # ImageDataGeneratorの生成
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
    
        
        
            g = datagen.flow(x, batch_size=32, save_to_dir=output_dir, save_prefix='img', save_format='jpg')
            for i in range(NOS):
                batch = g.next()
            
    

