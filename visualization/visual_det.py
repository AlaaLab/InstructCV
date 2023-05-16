# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description **  Script for making visulization examples of object detection.
# ---------------------------------------------------------------------------------

import json
import numpy as np
import cv2
import random


colors           = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128,0,128), (255,255,255),
                    (0,206,209), (205, 133, 63), (165,42,42), (255,128,0), (188,143,143),(128,128,0)]
colors_used      = random.choice(colors)
bbox_p           = "det_result/"
img_p            = "ori_img/"

with open('bbox.json','r',encoding='utf8')as fp:
    bbox_data = json.load(fp)['bbox']

num = len(bbox_data)
img = cv2.imread('zebra.jpg')
for i in range(num):
    point1 = np.int_(bbox_data[i][:2])
    point2 = np.int_(bbox_data[i][2:])

    img = cv2.rectangle(img, point1, point2, colors_used, 2)
    new_point1 = (point1[0], point1[1] - 20)
    new_point2 = (point1[0] + 70, point1[1])

    img = cv2.rectangle(img,new_point1, new_point2, colors_used, -1) #background
    news_point1 = (point1[0] + 5 ,point1[1] - 5)
    
    cv2.putText(img, 'zebra', news_point1, cv2.FONT_HERSHEY_TRIPLEX, 0.5, colors_used, 1)

cv2.imwrite('zebra_result.jpg',img)