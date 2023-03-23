import os
from fnmatch import fnmatch
import cv2
from PIL import Image

path = './image_pairs'
file_list = os.listdir(path)

for file_name in file_list:
    
    img_list = os.listdir(os.path.join(path, file_name))
    
    for img_name in img_list:
        
        if fnmatch(img_name, "*.json"):
            continue
        
        path_check = os.path.join(path, file_name, img_name)
        
        img = Image.open(path_check)
        check = len(img.split())
        if check == 2:
            print(path_check)

        
    