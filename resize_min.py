# -*- coding: utf-8 -*-
"""
resize_min.py
大きすぎる画像を小さくするプログラム
Created on Wed Jun 12 14:53:42 2019

@author: hanano
"""


from PIL import Image
import os
import glob

#魚種ごとに画像を読み込む(フォルダ名=学名)
filelist = ["LateolabraxJaponicus","LateolabraxLatus","LateolabraxMaculatus","ThunnusOrientalis","ThunnusAlbacares","ThunnusTonggol","PagrusMajor","EvynnisTumifrons","DentexHypselosomusBleeker"]
for dir in filelist:
    # 入力ディレクトリを作成
    input_dir = dir
    files = glob.glob(input_dir + '\*')
    print(dir)

    # 出力ディレクトリを作成
    resize_dir = dir + '_resize'
    if os.path.isdir(resize_dir) == False:
        os.mkdir(resize_dir)
        
    n=1   
    for file in files:  
        img = Image.open(file)
        w,h = img.size
        #print('width: ', w)
        #print('height:', h)
        if(w > 1500 or h > 1500):
            img.thumbnail((1000, 1000), Image.ANTIALIAS)

        img.save('%s\%i.jpg'%(resize_dir,n))
        n+=1
