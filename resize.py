# -*- coding: utf-8 -*-
"""
resize.py
アスペクト比そのままで指定サイズ×指定サイズまでリサイズ
余白は白で埋める
Created on Tue Jun 18 14:39:53 2019

@author: hanano
"""


from PIL import Image
import os
import glob

class image_aspect():

    def __init__(self, image_file, aspect_width, aspect_height):
        self.img = Image.open(image_file)
        self.aspect_width = aspect_width
        self.aspect_height = aspect_height
        self.result_image = None

    def change_aspect_rate(self):

        img_width = self.img.size[0]
        img_height = self.img.size[1]




        if img_width > img_height:
            rate = self.aspect_width / img_width
        else:
            rate = self.aspect_height / img_height
#            print("rate = " + str(rate))

#        rate = round(rate, 1) # rateが四捨五入されるので、変な空白ができる原因
#        print("rate2 = " + str(rate))
        self.img = self.img.resize((int(img_width * rate), int(img_height * rate)))
        return self

    def past_background(self):
        self.result_image = Image.new("RGB", [self.aspect_width, self.aspect_height], (255, 255, 255))
        self.result_image.paste(self.img, (int((self.aspect_width - self.img.size[0]) / 2), int((self.aspect_height - self.img.size[1]) / 2)))
        return self

    def save_result(self, file_name):
        self.result_image.save(file_name)

filelist = ["LateolabraxJaponicus_resize","LateolabraxLatus_resize","LateolabraxMaculatus_resize","ThunnusOrientalis_resize","ThunnusAlbacares_resize","ThunnusTonggol_resize","PagrusMajor_resize","EvynnisTumifrons_resize","DentexHypselosomusBleeker_resize"]
#filelist = ["LateolabraxMaculatus_test"]
for dir in filelist:
    
    # 入力ディレクトリを作成
    input_dir = dir
    files = glob.glob(input_dir + '\*')
    print(dir)

    # 出力ディレクトリを作成
    resize_dir = dir + '_224'
    if os.path.isdir(resize_dir) == False:
        os.mkdir(resize_dir)
        
        
    n=1
    for file in files:  
        print(file)
        if __name__ == "__main__":
            image_aspect(file, 224, 224)\
                .change_aspect_rate()\
                .past_background()\
                .save_result("%s\%i.jpg"%(resize_dir,n))
        n+=1
