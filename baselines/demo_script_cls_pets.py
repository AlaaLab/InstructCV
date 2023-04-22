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

CLASS = ['Abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle','Bengal',
        'Birman', 'Bombay', 'boxer', 'British Shorthair', 'chihuahua', 'Egyptian Mau', 'english cocker spaniel',
        'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin',
        'keeshond', 'leonberger', 'Maine Coon', 'miniature pinscher', 'newfoundland', 'Persian',
        'pomeranian', 'pug', 'Ragdoll', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier',
        'shiba inu', 'Siamese', 'Sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']

def load_image_from_url(url):
    with urllib.request.urlopen(url) as f:
        img = Image.open(f)
        return np.array(img)

def main():

    model = runner.ModelRunner("xl", "/lustre/grp/gyqlab/lism/brt/language-vision-interface/baselines/unified-io-inference/xl.bin")
    
    split_path              = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/oxford-pets/annotations/test.txt'
    img_root                = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/oxford-pets/images'
    
    acc, count              = 0, 0
    
    
    for line in open(split_path):
        
        line                = line.strip()
        img_id              = line.split(' ')[0] #Abyssinian_201
        
        img_path            = os.path.join(img_root, img_id+".jpg")
        image               = Image.open(img_path)
        label               = ' '.join(img_id.split('_')[:-1]).strip()

        if len(image.size) == 3 and image.size[-1] == 3:
            continue
        
        image               = np.array(image)
        
        out     = model.image_classification(image, answer_options=['Abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle','Bengal',
                                                                    'Birman', 'Bombay', 'boxer', 'British Shorthair', 'chihuahua', 'Egyptian Mau', 'english cocker spaniel',
                                                                    'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin',
                                                                    'keeshond', 'leonberger', 'Maine Coon', 'miniature pinscher', 'newfoundland', 'Persian',
                                                                    'pomeranian', 'pug', 'Ragdoll', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier',
                                                                    'shiba inu', 'Siamese', 'Sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier'])

        
        if fnmatch (out['text'], label):
            acc += 1
            print("prediction is right!")
        count += 1
        print("output:{} / label:{}".format(out['text'], label))
        print("acc/count:{}/{}".format(acc,count))
        
    print("acc rate:{}".format(acc/count))



if __name__ == "__main__":
  main()