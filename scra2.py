# -*- coding: utf-8 -*-
"""
scra2.py (web魚図鑑用)　URLを取得
Created on Thu Jun  6 23:16:15 2019

@author: hanano
"""

import requests
import re
import uuid
from bs4 import BeautifulSoup
from six.moves import urllib

for num in range(1,3):
    r = requests.get('https://zukan.com/fish/internal230' + '?page=' + str(num))
    soup = BeautifulSoup(r.text,'lxml')
    imgs = soup.find_all('img',src=re.compile('/media/leaf/original/'))
    for img in imgs:

            print('https://zukan.com' +  img['src'].replace('?width=360&height=135&type=resize', ''))
  
