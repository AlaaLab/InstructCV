import os
import numpy as np
import cv2
import pdb
from PIL import Image
from fnmatch import fnmatch

CAT = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
       "Russian Blue", "Siamese", "Sphynx"}

DOG = {"american bulldog", "american pit bull terrier", "basset hound", "beagle", "Bengal", "boxer", "chihuahua", "english cocker spaniel", 
       "english setter", "german shorthaired", "great pyrenees", "havanese", "japanese chin", "keeshond", "leonberger", "miniature pinscher",
       "newfoundland", "pomeranian", "pug", "saint bernard", "samoyed", "scottish terrier", "shiba inu", "staffordshire bull terrier",
       "wheaten terrier", "yorkshire terrier"}

root                = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/img_test_cls_rp'

file_n              = os.listdir(root)

true = 0
all = 0

for imgs in file_n: # Abyssinian_2
    label           = ""
    gt              = 999
    for i in range(len(imgs.split("_"))-1):
        label       = label + " " + imgs.split("_")[i]
    label           = label.strip()
    
    if label in CAT:
        gt      = "cat"
    if label in DOG:
        gt      = "dog"
        
    imgs_p          = os.path.join(root, imgs)
    img_c           = os.listdir(imgs_p)
     
    img1_p      = os.path.join(imgs_p, img_c[0])
    img1_id     = img_c[0].split("_")[-1].split(".")[0]
    img2_p      = os.path.join(imgs_p, img_c[1])
    img2_id     = img_c[1].split("_")[-1].split(".")[0]
        
    img1            = Image.open(img1_p)
    img2            = Image.open(img2_p)
    img1_arr        = np.array(img1)   
    img2_arr        = np.array(img2)
    img1_dist       = np.sqrt(np.sum((img1_arr - np.array([0, 255, 0]))**2, axis=2))
    img2_dist       = np.sqrt(np.sum((img2_arr - np.array([0, 255, 0]))**2, axis=2))
    img1_mean_dist  = np.mean(img1_dist)
    img2_mean_dist  = np.mean(img2_dist)
    
    if abs(img1_mean_dist - img2_mean_dist) <= 2.2:
        true += 1
        all += 1
        continue
        
    min_dist        = min(img1_mean_dist, img2_mean_dist)
    
    if min_dist     == img1_mean_dist:
        #prediction: img1
        pred        = img1_id
        if pred == gt:
            true += 1

    elif min_dist   == img2_mean_dist:
        #prediction: img2
        pred        = img2_id
        if pred == gt:
            true += 1
            
    all += 1
acc = true / all
print("True/ALL: {}/{}. pred/label:{}/{}".format(true, all, pred, gt))
print("acc:", acc)
        




    
        
        