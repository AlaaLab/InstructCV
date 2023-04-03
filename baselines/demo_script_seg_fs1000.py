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

  model = runner.ModelRunner("xl", "/lustre/grp/gyqlab/lism/brt/language-vision-interface/baselines/unified-io-inference/xl.bin")
  
  fs1000_root               = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/fewshot_data/fewshot_data"
  save_root                 = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/imgs_test_fs1000_bsln"
  task                      = "fs_seg"
  
  for file_path in open(os.path.join(fs1000_root,"test_part0.txt")):# neck_brace/10.jpg
      
      file_path               = file_path.strip()
      img_name                = file_path.split("/")[1]
      cls_name                = file_path.split("/")[0]
      id                      = img_name.split(".")[0]

      ge_path                 = os.path.join(save_root, cls_name + "_" + id + "_" + task)
      gt_save_path            = os.path.join(ge_path, cls_name + "_" + id + "_" + task + "_gt.jpg")
      pred_save_path          = os.path.join(ge_path, cls_name + "_" + id + "_" + task + "_pred.jpg")
        
      if os.path.exists(ge_path) == False:
          os.makedirs(ge_path)
        
      if fnmatch(file_path, "*jpg"):
  
          img_path = os.path.join(fs1000_root, file_path)
        
      if fnmatch(file_path, "*png"):
            
          shutil.copy(os.path.join(fs1000_root, file_path), gt_save_path)
          continue

      prompts             = "What is the segmentation of \"*\" ?"
      cname               = cls_name.replace("_"," ")
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