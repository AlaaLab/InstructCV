# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Calculate mAR、mAP metrics for object detection
# --------------------------------------------------------
# ** Reference ** 
# cocoapi: https://github.com/cocodataset/cocoapi
# --------------------------------------------------------

import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Eval():
    CLASSES = ('book','chair')  #can determine which CLASSES to test AR for by adjusting classes

    def __init__(self):
        pass

    def evaluate(self,
                 ann_file = 'ann.json',
                 pred_file = 'pred.json',
                 metric='bbox'):
        coco_gt = COCO(ann_file)
        self.cat_ids = coco_gt.getCatIds(catNms=self.CLASSES)
        self.img_ids = coco_gt.getImgIds()
        #print(self.cat_ids, self.img_ids)
        iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        
        iou_type = metric
        coco_det = coco_gt.loadRes(pred_file)

        cocoEval = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list((100,300,1000))
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

if __name__ == '__main__':
    eval_tools = Eval()
    eval_tools.evaluate()
