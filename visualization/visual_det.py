# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description **  Script for making visulization examples of object detection.
# ---------------------------------------------------------------------------------

import json
import numpy as np
import cv2
import random

colors           = [(252,230.202),(255,0,0),(255,127,80),(255,99,71),(255,0,255),(0,255,0),(0,255,255),(255,235,205),(255,255,0),(255,153,18),(255,215,0),(255,227,132),
                    (160,32,240),(244,164,95),(218,112,214),(153,51,250),(255,97,0),(106,90,205),(127,255,212),(255,125,64),(0,199,140),(3,168,158)]
bbox_p           = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/det_result/bbox_woody.json"
img_p            = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/ori_img/woody.jpg"
cls_name         = ['Spider Man']

with open(bbox_p,'r',encoding='utf8')as fp:
    bbox_data = json.load(fp)['bbox']

num = len(bbox_data)
img = cv2.imread(img_p)

for i in range(num):
    colors_used      = random.choice(colors)
    point1 = np.int_(bbox_data[i][:2])
    point2 = np.int_(bbox_data[i][2:])

    img = cv2.rectangle(img, point1, point2, colors_used, 10)
    new_point1 = (point1[0], point1[1] - 30)
    new_point2 = (point1[0] + 250, point1[1])

    img = cv2.rectangle(img,new_point1, new_point2, colors_used, -1) #background
    news_point1 = (point1[0] + 5 ,point1[1] - 5)
    print(cls_name[i])
    cv2.putText(img, cls_name[i], news_point1, cv2.FONT_ITALIC, 1, (0,0,0), 2)

cv2.imwrite('det_vis.jpg',img)