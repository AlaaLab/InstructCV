# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description **  Script for 1. removing borders in the instance annotation of the VOC.
#                               2. removing images only contain one color.
# ---------------------------------------------------------------------------------

import cv2
import numpy as np
import glob
import pdb
import os

data_p = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/VOCdevkit/VOC2012/'

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

img_paths = glob.glob(os.path.join(data_p,'SegmentationObject/*.png'))
mkdir(os.path.join(data_p, 'SegmentationObject_new'))

for img_path in img_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    color_n = np.unique(gray)
    # if len(color_n) <= 3:
    #     continue
    for i in range(h):
        for j in range(w):
            if thresh[i,j] == 255:
                img[i,j,:] = 0

    save_p = os.path.join(data_p, 'SegmentationObject_new') + '/' + img_path.split('/')[-1]
    cv2.imwrite(save_p, img)