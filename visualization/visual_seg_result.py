# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Visualization tools for segmentation
# --------------------------------------------------------

import cv2
import numpy as np
import glob
img1 = cv2.imread('./data_2/img1.jpg')
img2 = img1.copy()
img_paths = glob.glob('./data_2/*')
img_paths.pop(-1)
np.random.seed(2)
colors = np.random.randint(0,255,(len(img_paths),3))
np.random.seed(5)
colors2 = np.random.randint(0,255,(len(img_paths),3))
for i in range(len(img_paths)):
    img_path = img_paths[i]
    img = cv2.imread(img_path,0)
    contours,_ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    img_1 = cv2.drawContours(img1,contours,-1,(int(colors[i,0]),int(colors[i,1]),int(colors[i,2])),3)
    # point = np.array(contours)
    img_2 = cv2.fillPoly(img2, [contours[0]], (int(colors2[i,0]),int(colors2[i,1]),int(colors2[i,2])))

img_add = cv2.addWeighted(img_1, 0.7,img_2, 0.3, 0)
cv2.imshow('result',img_add)
cv2.waitKey(0)
cv2.destroyAllWindows()