# Copyright Yulu Gan
# modifed from unified-io, 29 Mar
# support: evaluate model's performance on nyuv2

import argparse
from os.path import exists
from PIL import Image
from uio import runner
from uio.configs import CONFIGS
from fnmatch import fnmatch
import numpy as np
from absl import logging
import warnings
import os
import time
from fnmatch import fnmatch
import shutil
import cv2
import urllib
import pdb
import sys
sys.path.append("././dataset_creation")
from format_dataset import preproc_coco

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def load_image_from_url(url):
    with urllib.request.urlopen(url) as f:
        img = Image.open(f)
        return np.array(img)

def main():

    model = runner.ModelRunner("xl", "/lustre/grp/gyqlab/lism/brt/language-vision-interface/baselines/unified-io-inference/xl.bin")
    
    split_path              = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/coco/test_part0.txt'
    img_root                = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/coco/val2017'
    coco_root               = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/coco'
    
    img_info, clses         = preproc_coco(coco_root)
    
    acc, count              = 0, 0
    
    
    for line in open(split_path):
        start = time.time()
        line                = line.strip() #000000355677.jpg
        img_id              = line.split('.')[0].lstrip("0") #355677
        cname_id            = list(img_info[str(img_id)].keys())
        ncls_perimg          = [] # store g.t class names
        for i in cname_id:
            cname           = clses[i]
            ncls_perimg.append(cname)
            
        img_path            = os.path.join(img_root, line)
        image               = Image.open(img_path)

        if len(image.size) == 3 and image.size[-1] == 3:
            continue
        
        image               = np.array(image)
        
        out     = model.image_classification(image, answer_options=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                                                                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                                                                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                                                                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                                                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                                                                    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                                                                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

        for i in range(len(ncls_perimg)):
            if fnmatch (out['text'], ncls_perimg[i]):
                acc += 1
                print("prediction is right!")
        count += 1
        print("acc/count:{}/{}".format(acc,count))
    print("acc rate:{}".format(acc/count))



if __name__ == "__main__":
  main()