# -*- coding: utf-8 -*-
"""
scra.py 
クローリング+スクレイピングを行うプログラム
Created on Thu Jun  6 23:16:15 2019

@author: hanano
"""

import requests
import re
import uuid
from bs4 import BeautifulSoup
from six.moves import urllib

#取得したい魚の和名を入力
keyword = urllib.parse.quote('キダイ',encoding='shift-jis')

for num in range(1,40,20): #range(min_num,max_num,間隔)

    r = requests.get('http://fishpix.kahaku.go.jp/fishimage/search?START=' + str(num) + '&JPN_FAMILY=&FAMILY=&JPN_NAME=' + keyword + '&SPECIES=&LOCALITY=&FISH_Y=&FISH_M=&FISH_D=&PERSON=&PHOTO_ID=&JPN_FAMILY_OPT=1&FAMILY_OPT=1&JPN_NAME_OPT=0&SPECIES_OPT=1&LOCALITY_OPT=1&PERSON_OPT=1&PHOTO_ID_OPT=1')
    soup = BeautifulSoup(r.text,'lxml')
    imgs = soup.find_all('img',src=re.compile('../photos/'))

    for img in imgs:
        print('http://fishpix.kahaku.go.jp' +   img['src'].lstrip('..').replace('AI', 'AF'))
        r = requests.get('http://fishpix.kahaku.go.jp' +   img['src'].lstrip('..').replace('AI', 'AF'))    
        with open(str('C:\\Users\\eupho\\.spyder-py3\\tmp/')+str(uuid.uuid4())+str('.jpeg'),'wb') as file:
            file.write(r.content)
