import os
import numpy as np
import cv2
import pdb
from fnmatch import fnmatch

root = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/img_test_cls_petss'

file_list = os.listdir(root)

acc                 = 0
con                 = 0
true                = 0
mean_clses_all      = {}
label               = ""

for file in file_list:
    file_path       = os.path.join(root,file)
    img_list        = os.listdir(file_path)
    label           = ""

    for i in range(len(file.split("_")) - 1):
        label = label + " " +file.split("_")[i]
    
    mean_clses = {}
    for img_name in img_list:
        img_path    = os.path.join(file_path,img_name)
        image       = cv2.imread(img_path)
        img_name    = img_name.split("cls_")[1]
        img_name    = img_name.split(".")[0]
        mean        = np.mean(image)
        mean_clses[img_name] = mean
        prediction = max(mean_clses,key=mean_clses.get)
    print("pred:{} | label:{}".format(prediction, label))
    if fnmatch(prediction, label):
        true += 1
    con += 1
    acc = true/con
    print("acc:{} | con:{}".format(acc, con))
        
    mean_clses_all[file] = mean_clses

# print(mean_clses_all)
    
        
        