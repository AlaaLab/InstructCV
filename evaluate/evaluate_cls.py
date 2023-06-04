import os
import numpy as np
import cv2
import pdb
from PIL import Image
from fnmatch import fnmatch

root                = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/img_test_cls_rp'

file_n              = os.listdir(root)

for imgs in file_n: # Abyssinian_2
    imgs_p          = os.path.join(root, imgs)
    img_c           = os.listdir(imgs_p)
     
    img1_p      = os.path.join(imgs_p, img_c[0])
    img1_id     = img_c[0].split("_")[-1].split(".")[0]
    img2_p      = os.path.join(imgs_p, img_c[1])
    img2_id     = img_c[1].split("_")[-1].split(".")[0]
    print("img2_id:", img2_id)
        
        
    img1            = Image.open(img1_p)
    img2            = Image.open(img2_p)
    img1_arr        = np.array(img1)   
    img2_arr        = np.array(img2)
    img1_dist       = np.sqrt(np.sum((img1_arr - np.array([255, 0, 0]))**2, axis=2))
    img2_dist       = np.sqrt(np.sum((img2_arr - np.array([255, 0, 0]))**2, axis=2))
    img1_mean_dist  = np.mean(img1_dist)
    img2_mean_dist  = np.mean(img2_dist)
    min_dist        = min(img1_mean_dist, img2_mean_dist)
    
    if min_dist     == img1_mean_dist:
        print('1.jpg离红色最近')
    elif min_dist   == img2_mean_dist:
        print('2.jpg离红色最近')
        




    
        
        