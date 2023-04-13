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

  model = runner.ModelRunner("xl", "/fsx/proj-lvi/language-vision-interface/baselines/unified-io-inference/xl_1000k.bin")
  
  root            = "/fsx/proj-lvi/language-vision-interface/data/nyuv2_mdet/nyu"
  save_root       = "/fsx/proj-lvi/language-vision-interface/outputs/img_pairs_eva_dep"


  for line in open(os.path.join(root, 'nyu_test.txt')):

      start       = time.time()
      
      line        = line.strip()
      img_name    = line.split(' ')[0]
      gt_name     = line.split(' ')[1]
      img_id      = img_name.split('.')[0].replace("/","_")
      gt_id       = gt_name.split('.')[0].replace("/","_")
      img_path    = os.path.join(root, img_name)
      gt_path     = os.path.join(root, gt_name)
      # gt_img      = cv2.imread(gt_path)
      
      with Image.open(img_path) as img:
        image = np.array(img.convert('RGB'))

        out = model.run([image], ["What is the depth map of the image ?"], 
                  output_text_len=1, generate_image=True, num_decodes=None)
        depth_image = out["image"][0]

        save_img_path = os.path.join(save_root, img_id)
        
        if os.path.exists(save_img_path) == False:
          os.makedirs(save_img_path)
        
        if os.path.exists(save_img_path + "/{}_pred.png".format(img_id)) == False:
          assert "path not exist!"
        
        # cv2.imwrite(save_img_path + "/{}_pred.png".format(img_id), depth_image)
        if os.path.isfile(gt_path):
            shutil.copy(gt_path,save_img_path + "/{}_gt2.png".format(gt_id))
            
        # cv2.imwrite(save_img_path + "/{}_gt.png".format(gt_id), gt_img)
        
        end = time.time()
        
        print("one image done, cost time:{}".format(end-start))


if __name__ == "__main__":
  main()