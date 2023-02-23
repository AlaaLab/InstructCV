"coding = utf-8"

import shutil
import os,sys
import cv2
import pdb
from PIL import Image
import numpy as np
from os import path

input_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/oxford-pets/image_pairs'

import os
 
def get_file_path_by_name(file_dir):

    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
    # print('Total Number', len(L))
    return L
 
path_ls = get_file_path_by_name(input_path)

for item in path_ls:  
          
    img_path = os.path.join(input_path, item)
    
    image = Image.open(img_path)
    image = np.array(image)
    if image.shape[2] != 3:
        print(item)
        print('image.shape',image.shape)
