# -*- coding: utf-8 -*-
"""
pic_url_scra.py
画像のURLを取得する
Created on Thu Jun  6 23:16:15 2019

@author: hanano
"""

import requests
import re
import uuid
from bs4 import BeautifulSoup


url = "https://zukan.com/fish/"

#r = requests.get(url%(keyword))
#soup = BeautifulSoup(r.text,'lxml')
#imgs = soup.find_all('img',src=re.compile('^https://msp.c.yimg.jp/yjimage'))
#for img in imgs:
#        print(img['src'])
