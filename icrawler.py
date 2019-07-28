"""
icrawler.py
icrawlerを使って画像をとってくる
Created on Thu Jun  6 22:42:23 2019

@author: hanano
"""

from icrawler.builtin import GoogleImageCrawler
import sys
import os
argv = sys.argv

#取得したい画像の名前を入れる　ここではスズキの学名
if not os.path.isdir("Dentex hypselosomus Bleeker"):
    os.makedirs("Dentex hypselosomus Bleeker")

crawler = GoogleImageCrawler(storage = {"root_dir" : "Dentex hypselosomus Bleeker"})
crawler.crawl(keyword = "Dentex hypselosomus Bleeker", max_num = 300)
