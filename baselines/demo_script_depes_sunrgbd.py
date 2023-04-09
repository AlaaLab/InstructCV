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
import shutil
import cv2

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def main():

  model = runner.ModelRunner("xl", "/lustre/grp/gyqlab/lism/brt/language-vision-interface/baselines/unified-io-inference/xl.bin")
  
  root            = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/"
  save_root       = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/imgs_test_sunrgbd_unifiedio"

  with open(os.path.join(root, 'SUNRGBD/SUNRGBD_val_splits4.txt')) as file: 

      for line in file:
          start = time.time()
          img_path_part   = line.strip().split(" ")[0] # SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg
          file_name       = img_path_part.split("/")[-4] # kinect2data
          img_name        = img_path_part.split("/")[-1] # 0000103.jpg
          img_id          = file_name + "_" + img_name.split(".")[0] # kinect2data_0000103
          gt_path_part    = line.strip().split(" ")[1]
          gt_path         = os.path.join(root, gt_path_part)
          gt_id           = gt_path_part.split("/")[-4] + "_" + gt_path_part.split("/")[-1].split(".")[0]

      
          with Image.open(os.path.join(root,img_path_part)) as img:
            image = np.array(img.convert('RGB'))

            out = model.run([image], ["What is the depth map of the image ?"], 
                      output_text_len=1, generate_image=True, num_decodes=None)
            depth_image = out["image"][0]

            save_img_path = os.path.join(save_root, img_id)
            
            if os.path.exists(save_img_path) == False:
              os.makedirs(save_img_path)
            
            cv2.imwrite(save_img_path + "/{}_pred.png".format(img_id), depth_image)
            
            if os.path.isfile(gt_path):
                shutil.copy(gt_path,save_img_path + "/{}_gt.png".format(gt_id))
                
            # cv2.imwrite(save_img_path + "/{}_gt.png".format(gt_id), gt_img)
            
            end = time.time()
            
            print("one image done, cost time:{}".format(end-start))


if __name__ == "__main__":
  main()