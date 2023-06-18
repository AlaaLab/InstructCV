# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Visualization tools for segmentation
# --------------------------------------------------------

import cv2
import numpy as np
import glob
import pdb

img1 = cv2.imread('/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/ori_img/ADE_val_00000455.jpg')
h, w ,c = img1.shape
print(img1.shape)
img2 = img1.copy()

img_paths = glob.glob('/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/seg_result2/*')
save_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/123.jpg'

np.random.seed(5)
# colors = np.random.randint(0,255,(len(img_paths),3))
colors = np.random.randint(0,255,(1,3))
np.random.seed(6)
# colors2 = np.random.randint(0,255,(len(img_paths),3))
colors2 = np.random.randint(0,255,(1,3))

print("len(img_paths)", len(img_paths))
for i in range(len(img_paths)):
    
    img_path = img_paths[i]
    img = cv2.imread(img_path,0)
    img = cv2.resize(img, (w,h))
    print(img.shape)
    contours,_ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    img_1 = cv2.drawContours(img1,contours,-1,(int(colors[i,0]),int(colors[i,1]),int(colors[i,2])),3)
    # img_1 = cv2.drawContours(img1,contours,-1,(0,255,0),3)
    for j in range(len(contours)):
        img_2 = cv2.fillPoly(img2, [contours[j]], (int(colors2[i,0]),int(colors2[i,1]),int(colors2[i,2])))
        # img_2 = cv2.fillPoly(img2, [contours[j]], (64,255,64))

img_add = cv2.addWeighted(img_1, 0.5,img_2, 0.5, 0)
cv2.imwrite(save_path, img_add)