from PIL import Image
import os
import pdb
import cv2
import numpy as np
from fnmatch import fnmatch

nyu_root = "./data/nyu_mdet"
file_list = os.listdir(nyu_root)

for file_name in file_list: # file_name: basement_0001a
    
    if fnmatch(file_name, '*.txt'):
        continue
    
    img_list = os.listdir(os.path.join(nyu_root, file_name))
    
    for img_name in img_list: # img_name: rgb_00000.jpg / sync_depth_00000.png
        
        save_path   = os.path.join(nyu_root, file_name, "vli_depth_" + "".join(img_name.split("_")[-1]))
        print(save_path)
        
        if fnmatch(img_name, 'vli_depth_*'):
            continue
        
        if fnmatch(img_name, '*.jpg'):
        
            img_path    = os.path.join(nyu_root, file_name, img_name)
        
        if fnmatch(img_name, '*.png'):
            
            depth_path  = os.path.join(nyu_root, file_name, img_name)
        
        
            depth_img   = cv2.imread(depth_path, 0)
            print(depth_img.shape)
            print(depth_img.dtype)
            depth_img   = depth_img.astype(np.float32)
            depth_img   = depth_img / 1000 * 255 / 10
            # cv2.normalize(depth_img, alpha=0, beta=255, normType=cv2.min)
            depth_img   = depth_img.astype(np.uint8)
            depth_img2        = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
            pdb.set_trace()
            cv2.imwrite(save_path, depth_img2)
            
            
            # used for cheking
            # print(np.sum(img2[:,:,0] == img))
            # print(np.sum(img2[:,:,0] == img2[:,:,1]))
            # print(np.sum(img2[:,:,0] == img2[:,:,2]))
            # print(img2[:,:,2] == img)

