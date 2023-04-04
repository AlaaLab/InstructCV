# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Generating the pred.json for calculating metrics
# --------------------------------------------------------


import os
import json
#settings
dst_file_name               = './pred.json'      #save path
src_dir                     = './coco_det'       #path to cocodet
src_file_item               = 'pred_bbox.json'   #item name
bbox_item                   = 'pred_bbox'        #key name in json file


#For write
json_results                = []

imgname_list                = []
clsname_list                = []

bbox_id = 0
files = os.listdir(src_dir)
for file in files:
    if '.' in file:
        continue
    with open(os.path.join(src_dir, file, src_file_item),'r') as fr:
        imgname, clsname, _ = file.split('_')
        #add image info.
        if imgname not in imgname_list:
            imgname_list.append(imgname)
            img_id          = len(imgname_list)-1
        #add cls info.
        if clsname not in clsname_list:
            clsname_list.append(clsname)
            ann_id          = len(clsname_list)-1

        img_id              = imgname_list.index(imgname)
        ann_id              = clsname_list.index(clsname)
        #add ann info.
        bboxes = json.load(fr)[bbox_item]
        for bbox in bboxes:
            x1,y1,x3,y3     = bbox
            w               = x3 - x1
            h               = y3 - y1
            area            = w*h 

            json_results.append({
                "image_id": img_id,                             # int 该物体所在图片的编号
                "category_id": ann_id,                          # int 被标记物体的类别id编号
                "bbox": [x1, y1, w, h],                         # [x, y, width, height] 目标检测框的坐标信息
                "score": 1
            })

json.dump(json_results,open(dst_file_name,'w'))


        

