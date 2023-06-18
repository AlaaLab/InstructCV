# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Calculate mAR、mAP metrics for object detection
# --------------------------------------------------------
# ** Reference ** 
# cocoapi: https://github.com/cocodataset/cocoapi
# --------------------------------------------------------

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append("./")
from evaluate.evaluate_cls_seg_det import cal_bboxes_iou
from dataset_creation.format_dataset import CLASSES_COCO
#can determine which CLASSES to test AR for by adjusting classes_coco

class Eval():

    def __init__(self, dst_pred='./pred_final.json', dst_ann='./ann_final.json',
                 pred_file= 'pred_final.json', ann_file='ann_final.json',
                 src_ann_item='bbox.json', src_pred_item='pred_bbox2.json',
                 ann_bbox_item='bbox', pred_bbox_item='pred_bbox', 
                 src_dir='./outputs/imgs_test_voc_det_rp',metric='bbox'):
        
        self.dst_pred                       = dst_pred
        self.dst_ann                        = dst_ann
        self.src_dir                        = src_dir
        self.src_pred_item                  = src_pred_item
        self.src_ann_item                   = src_ann_item
        self.pred_bbox_item                 = pred_bbox_item
        self.ann_bbox_item                  = ann_bbox_item
        self.ann_file                       = ann_file
        self.pred_file                      = pred_file
        self.metric                         = metric
        self.CLASSES_COCO                   = CLASSES_COCO

    def deal_annojson(self):
        
        if os.path.exists(self.dst_ann) == True:
            return None
        #For write
        dst_datas                           = {    
                                                "images" : [],
                                                "annotations" : [],
                                                "categories" : []
                                                }

        imgname_list, clsname_list          = [],[]
        bbox_id                     = 0
        files                       = os.listdir(self.src_dir)

        for file in files:
            if '.' in file:
                continue
            
            with open(os.path.join(self.src_dir, file, self.src_ann_item),'r') as fr:
                year, imgid, clsname, _ = file.split('_')
                #add image info.
                imgname = year + "_" + imgid
                if imgname not in imgname_list:
                    imgname_list.append(imgname)
                    img_id = len(imgname_list)-1
                    dst_datas['images'].append({
                        "id": img_id,                               # int image id. Can start from 0
                        "file_name": "{}.jpg".format(img_id),       # str file name.
                        "width": 512,                               # int image weight.
                        "height": 512,                              # int image height.
                    })
                 
                #add cls info.
                if clsname not in clsname_list:
                    clsname_list.append(clsname)
                    ann_id          = len(clsname_list)-1
                    dst_datas['categories'].append({
                        "id": ann_id,
                        "name": clsname,
                        "supercategory": "None"
                    })

                img_id              = imgname_list.index(imgname)
                ann_id              = clsname_list.index(clsname)
                #add ann. info.
                bboxes              = json.load(fr)[self.ann_bbox_item]
                
                for bbox in bboxes:
                    x1,y1,x3,y3     = bbox
                    w               = x3 - x1
                    h               = y3 - y1
                    area            = w*h 

                    dst_datas["annotations"].append({
                        "id": bbox_id,                               # int 图片中每个被标记物体的id编号
                        "image_id": img_id,                          # int 该物体所在图片的编号
                        "category_id": ann_id,                       # int 被标记物体的类别id编号
                        "iscrowd": 0,                                # 0 or 1 目标是否被遮盖，默认为0
                        "area": area,                                # float 被检测物体的面积（64 * 64 = 4096)
                        "bbox": [x1, y1, w, h],                      # [x, y, width, height] 目标检测框的坐标信息
                        "segmentation": [[x1, y1, x3, y1, x3, y3, x1, y3]] 
                    })
                    
                    bbox_id        += 1

        json.dump(dst_datas,open(self.dst_ann,'w'))
        return bboxes

    def deal_predjson(self):
        if os.path.exists(self.dst_pred) == True:
            return None
        
        #For write
        json_results, imgname_list, clsname_list   = [],[],[]
        bbox_id = 0
        
        files = os.listdir(self.src_dir)
        for file in files:
            if '.' in file:
                continue
            
            with open(os.path.join(self.src_dir, file, self.src_pred_item),'r') as fr:
                year, imgid, clsname, _ = file.split('_')
                #add image info.
                imgname = year + "_" + imgid
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
                bboxes = json.load(fr)[self.pred_bbox_item]
                for bbox in bboxes:
                    x1,y1,x3,y3     = bbox
                    w               = x3 - x1
                    h               = y3 - y1
                    area            = w*h 

                    json_results.append({
                        "image_id": img_id,                             # int
                        "category_id": ann_id,                          # int
                        "bbox": [x1, y1, w, h],                         # [x, y, width, height]
                        "score": 1
                    })
        json.dump(json_results,open(self.dst_pred,'w'))
        return bboxes


    def evaluate(self):
        
        pred_bboxes_            = cal_bboxes_iou(self.src_dir, self.src_ann_item,
                                                 self.src_pred_item, self.ann_bbox_item, self.pred_bbox_item)
        bboxes_                 = self.deal_predjson()
        bboxes__                = self.deal_annojson()
        coco_gt                 = COCO(self.ann_file)
        self.cat_ids            = coco_gt.getCatIds(catNms=self.CLASSES_COCO)
        self.img_ids            = coco_gt.getImgIds()
        #print(self.cat_ids, self.img_ids)
        iou_thrs                = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        
        iou_type                = self.metric
        coco_det                = coco_gt.loadRes(self.pred_file)

        cocoEval                = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds  = self.cat_ids
        cocoEval.params.imgIds  = self.img_ids
        cocoEval.params.maxDets = list((100,300,1000))
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return


if __name__ == '__main__':
    eval_tools = Eval()
    eval_tools.evaluate()
