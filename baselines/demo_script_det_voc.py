# Copyright Yulu Gan
# modifed from unified-io, 29 Mar
# support: evaluate model's performance on nyuv2

import argparse
from os.path import exists
from PIL import Image
from uio import runner
from uio.configs import CONFIGS
import numpy as np
from absl import logging
import warnings
import os
import time
from fnmatch import fnmatch
import shutil
import cv2
import sys
sys.path.append("././")
from evaluate.evaluate_cls_seg_det import genGT, COLOR_VOC, CLASSES_VOC

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def get_colors(image_path):
    
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    colors = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            colors.add((r[i][j], g[i][j], b[i][j]))
    return list(colors)

def main():

  model = runner.ModelRunner("xl", "/lustre/grp/gyqlab/lism/brt/language-vision-interface/baselines/unified-io-inference/xl.bin")
  
  voc_root                  = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/VOCdevkit/VOC2012"
  save_root                 = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/imgs_test_voc_bsln"
  split                     = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/VOCdevkit/VOC2012/val_part8.txt"
  task                      = "voc_seg"
  
  genGT(voc_root, save_root, task, split).generate_voc_gt()
  
  for line in open(os.path.join(voc_root, split)):
    
      line = line.strip()
      img_path_part = line.split(" ")[0]
      gt_path_part  = line.split(" ")[1]
      img_name    = img_path_part.split("/")[1].split(".")[0]
      
      img_path                     = os.path.join(voc_root, img_path_part)
      seg_path                     = os.path.join(voc_root, gt_path_part)
      
      colors = get_colors(seg_path)
      
      for color in colors:
        idx = COLOR_VOC.index(list(color))
        if idx == 0 or idx == 21:
            continue
        cname = CLASSES_VOC[idx]
        
        img_name = seg_path.split(".")[0].split("/")[-1]
        out_name = img_name + "_" + cname + "_" + "seg"
        output_path = os.path.join(save_root, out_name)
        pred_save_path = output_path + '/{}_pred.png'.format(out_name)

        prompts             = "What is the segmentation of \"*\" ?"
        prompts             = prompts.replace("*", cname)
        print("prompt:", prompts)
        print("seg_path:", seg_path)
        start               = time.time()
        
        with Image.open(img_path) as img:
            image = np.array(img.convert('RGB'))
          
            out   = model.object_segmentation(image, cname)
            
            if len(out["mask"]) == 0 or out["mask"] is []:
              img = np.zeros((256, 256, 3), np.uint8)
              cv2.imwrite(pred_save_path, img)
              continue
            
            depth_image = np.expand_dims(np.stack(out["mask"]), -1)
            depth_image = depth_image[0].squeeze()
            
            depth_image = Image.fromarray(depth_image)
            depth_image.save(pred_save_path)
            # cv2.imwrite(pred_save_path, depth_image)

            end = time.time()
            
            print("one image done, cost time:{}".format(end - start))


if __name__ == "__main__":
  main()