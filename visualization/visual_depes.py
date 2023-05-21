import cv2
import numpy as np
import os
import pdb

root = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/result_dep'
save_r = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/output'

img_list = os.listdir(root)

for img in img_list:
    
    img_p = os.path.join(root, img)
    img_id = img.strip()
    
    img_gray = cv2.imread(img_p, 0)
    n_min    = np.min(img_gray)
    n_max    = np.max(img_gray)
    
    img_gray = (img_gray-n_min)/(n_max-n_min+1e-8)
    img_gray = (255*img_gray).astype(np.uint8)
    img_pseudo = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

    save_p = os.path.join(save_r, img)
    cv2.imwrite(save_p, img_pseudo)