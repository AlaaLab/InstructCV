from PIL import Image
import os
import pdb
import cv2
import numpy as np
from fnmatch import fnmatch

nyu_root = "./data/nyu_data/data/nyu2_test"
save_root = "./outputs/imgs_test_nyuv2_evaluate"

img_list = os.listdir(nyu_root)

for img_name in img_list: # img_name: 000000_depth.png
    
    
    if fnmatch(img_name, "*colors.png"):
        continue
    
    if fnmatch(img_name, 'vli_depth_*'):
        continue
    
    if fnmatch(img_name, '*vli_depth_*'):
        continue
    
    if fnmatch(img_name, '*.png'): # img_name: 00013_depth.png
        
        save_path_folder_gt     = os.path.join(save_root, img_name.split(".")[0])
        save_path_file_gt       = os.path.join(save_path_folder_gt, img_name.split("_")[0] + "_vli_depth_gt" + ".png")
        
        save_path_folder_pred   = os.path.join(save_root, img_name.split(".")[0])
        save_path_file_pred     = os.path.join(save_path_folder_pred, img_name.split("_")[0] + "_vli_depth_pred" + ".png")
        
        
        # generate processed gt
        depth_path          = os.path.join(nyu_root, img_name)
        depth_img           = cv2.imread(depth_path, -1)
        depth_img           = depth_img.astype(np.float32)
        depth_img           = depth_img / 1000 * 255 / 10
        depth_img           = depth_img.astype(np.uint8)

        depth_img_gt        = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
        
        # open img_pred
        img_pred_path       = "./outputs/imgs_test_nyuv2"
        img_name_           = img_name.split('.')[0].split('_')[0] #00013
        depth_img_pred      = cv2.imread(os.path.join(img_pred_path, img_name_ + "_colors_test_depth.jpg"))
        
        
        if os.path.exists(save_path_folder_gt) == False:
            os.makedirs(save_path_folder_gt)
        
        if os.path.exists(save_path_folder_pred) == False:
            os.makedirs(save_path_folder_pred)

        cv2.imwrite(save_path_file_gt, depth_img_gt) #gt
        cv2.imwrite(save_path_file_pred, depth_img_pred) #pred
        
        
        # used for cheking
        # print(np.sum(img2[:,:,0] == img))
        # print(np.sum(img2[:,:,0] == img2[:,:,1]))
        # print(np.sum(img2[:,:,0] == img2[:,:,2]))
        # print(img2[:,:,2] == img)

