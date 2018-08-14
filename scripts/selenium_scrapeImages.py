# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib.request as urllib2
import argparse


def scrapeImages(searchterm, number_images=20):
    # searchterm will also be the name of the folder

    print('Download __ ', number_images, ' __ of searchterm: ', searchterm)
    url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"

    webdriver_path = "tmp/chromedriver_win32/chromedriver.exe"
    print('webdriver-path: ', os.getcwd()+webdriver_path)
    browser = webdriver.Chrome(webdriver_path)

    browser.get(url)
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    counter = 0
    succounter = 0

    prefix_path = "./../data/2D/google_search_images/"

    if not os.path.exists(prefix_path+searchterm):
        try:
            os.makedirs(prefix_path+searchterm)
            print('scrapeImages: images successfully saved in: ', prefix_path+searchterm)
        except:
            print('scrapeImages: another problem occurs: ', prefix_path+searchterm)
    else:
        print('scrapeImages: propably the folder already exists: ', prefix_path+searchterm)

    for _ in range(500):
        browser.execute_script("window.scrollBy(0,10000)")

    for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
        if counter == number_images:
            break

        counter = counter + 1
        print("Total Count:", counter, "\tSuccsessful Count:", succounter)
        print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
        url = json.loads(x.get_attribute('innerHTML'))["ou"]

        img = json.loads(x.get_attribute('innerHTML'))["ou"]
        imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]

        try:
            req = urllib2.Request(img, headers={'User-Agent': header})
            # raw_img = urllib2.urlopen(req).read()
            raw_img = urllib2.urlopen(url).read()
            print(os.path.join(searchterm , searchterm + "_" + str(counter) + "." + imgtype))
            # filename = "{0}{1}/{1}_{2}.{3}".format(prefix_path, searchterm, str(counter), imgtype)
            filename = "{0}{1}/{1}_{2}.jpg".format(prefix_path, searchterm, str(counter))
            print(filename)
            File = open(filename, "wb")
            File.write(raw_img)
            File.close()
            succounter = succounter + 1
        except:
                print("can't get img")

    print(succounter, "pictures succesfully downloaded")
    browser.close()
