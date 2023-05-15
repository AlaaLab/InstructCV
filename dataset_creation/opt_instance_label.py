# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Make colored instance label for ade20k
# --------------------------------------------------------

import os
from PIL import Image
import numpy as np
import json
import random
import cv2

img_path = './data/ADEChallengeData2016/images/validation'
label_path = './data/ADEChallengeData2016/annotations_instance/validation'
out_path = './data/ADEChallengeData2016/annotations_instance_vis/validation'

colors = [(0,0,0)]

for i in range(255):
    colors.append((random.randrange(1, 255, 1), random.randrange(1, 255, 1), random.randrange(1, 255, 1)))

print(colors)
colors = np.array(colors)

n = 0


for img_p in os.listdir(img_path):
    
    img = np.array(Image.open(os.path.join(img_path, img_p)))
    label_p = os.path.join(label_path, img_p.split('.')[0] + '.png')
    im_label = Image.open(label_p).convert('L')
    w, h = im_label.size
    im_label = np.array(im_label)
    im_label = colors[im_label]
    new_img = np.where(im_label == 0, img[:,:], im_label[:,:])
    new_img = Image.fromarray(new_img.astype(np.uint8))
    # new_img.save(os.path.join(out_path, img_p.split('.')[0] + '.png'))
    # im_label.save(os.path.join(out_path, img_p.split('.')[0] + '.png'))
    cv2.imwrite(os.path.join(out_path, img_p.split('.')[0] + '.png'), im_label)
    n += 1
    if n % 100 == 0:
        print('{} samples processed!'.format(n))
