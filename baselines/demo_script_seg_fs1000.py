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

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def main():

  model = runner.ModelRunner("xl", "/fsx/proj-lvi/language-vision-interface/baselines/unified-io-inference/xl_1000k.bin")
  
  root            = "/fsx/proj-lvi/language-vision-interface/data/fss-1000/fewshot_data/fewshot_data"
  save_root       = "/fsx/proj-lvi/language-vision-interface/outputs/imgs_test_fs1000"
  task            = "seg_fs1000"
  
  for file_name in os.listdir(root): #e.g., file_name: abcus
    
    file_path = os.path.join(root, file_name)
    
    for img_name in os.listdir(file_path):
        
        img_id              = img_name.split(".")[0]
        root                = os.path.join(save_root, file_name + img_id)
        gt_save_path        = os.path.join(root, img_id + "_gt.jpg")
        pred_save_path      = os.path.join(root, img_id + "_pred.jpg")
        
        if os.path.exists(root) == False:
            os.makedirs(root)
        
        if fnmatch(img_name, "{}.jpg".format(img_id)):
    
            img_path = os.path.join(file_path, img_name)
        
        if fnmatch(img_name, "{}.png".format(img_id)):
            
            shutil.copy(os.path.join(file_path, img_name), gt_save_path)
            continue
        
        prompts             = "What is the segmentation of \"*\" ?"
        cname               = file_name.replace("_"," ")
        prompts             = prompts.replace("*", cname)
        print("prompt:", prompts)
        start               = time.time()
      
        with Image.open(img_path) as img:
          image = np.array(img.convert('RGB'))

          # out = model.run([image], [prompts], 
          #           output_text_len=1, generate_image=True, num_decodes=None)
          # depth_image = out["mask"][0]
          
          out   = model.object_segmentation(image, cname)
          depth_image = np.expand_dims(np.stack(out["mask"]), -1)
          depth_image = depth_image[0].squeeze()
        
          print("type:",type(depth_image))
          print("shape:",len(depth_image.shape))
          
          depth_image = Image.fromarray(depth_image)
          depth_image.save(pred_save_path)
          # cv2.imwrite(pred_save_path, depth_image)

          end = time.time()
          
          print("one image done, cost time:{}".format(end - start))


if __name__ == "__main__":
  main()