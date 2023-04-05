# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Evaluate the three tasks.
# --------------------------------------------------------
# References:
# mmdetection: https://github.com/open-mmlab/mmdetection
# cocoapi: https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
# --------------------------------------------------------

from argparse import ArgumentParser
import numpy as np
import cv2
import pdb
import os
import json
import random
from torchvision import transforms
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
import numpy as np
import math
from PIL import Image, ImageDraw
from dataset_creation.format_dataset import preproc_coco, get_bbox_img, CLASSES, get_seg_img



def iou(gt_img, pred_img):
    h, w, c = pred_img.shape

    gt_img = cv2.resize(gt_img, (w, h))

    predGray = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
    ret, predBinary = cv2.threshold(predGray, 127, 255, 0)

    gtGray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
    ret, gtBinary = cv2.threshold(gtGray, 127, 255, 0)

    inser = cv2.bitwise_and(predBinary, gtBinary)
    union = cv2.bitwise_and(predBinary, gtBinary)
    iou = inser.sum() / union.sum()
    print("iou:", iou)
    return iou


def cal_bboxes_iou(gt_bboxes, pred_bboxes):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """

    pred_bboxes_ = []

    for i in range(len(gt_bboxes)):
        
        iou_dict     = {}
        xmin1, ymin1, xmax1, ymax1 = gt_bboxes[i]
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        
        for j in range(len(pred_bboxes)):
            
            if len(pred_bboxes[j]) == 5:# add those are confident to be a rectangle
                pred_bboxes_.append(pred_bboxes[iou_dict[0][0]])
                continue
        
            xmin2, ymin2, xmax2, ymax2 = pred_bboxes[j]

            # Get the coordinates of the vertex of the intersection of rectangular boxes (intersection)
            xx1 = np.max([xmin1, xmin2])
            yy1 = np.max([ymin1, ymin2])
            xx2 = np.min([xmax1, xmax2])
            yy2 = np.min([ymax1, ymax2])

            area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1) # Calculate the area of two rectangular boxes
        
            inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # Calculate the intersection area
            
            # Calculate the iou
            iou = 0
            iou = inter_area / (area1 + area2 - inter_area + 1e-6)
            iou_dict[str(j)] = iou

        iou_dict = sorted(iou_dict.items(), key=lambda x: x[1], reverse=True)
        # pred_bboxes_.append(pred_bboxes[int(iou_dict[0][0])])
        # print("iou_dict", len(iou_dict))
        if len(iou_dict) == 0:
            return pred_bboxes_
        if iou_dict[0][1] > 0.5:
            pred_bboxes_.append(pred_bboxes[int(iou_dict[0][0])])
                
                
    return pred_bboxes_


def calc_iou(gt_img_path, pred_img):
    
    epsilon         = 1e-6
    h, w            = pred_img.size
    gt_img          = Image.open(gt_img_path)
    resize          = transforms.Resize([w,h])
    gt_img          = resize(gt_img)
    gt_img          = np.asarray(gt_img)
    
    intersection    = np.sum((gt_img) & (pred_img))
    union           = np.sum((gt_img) | (pred_img))

    iou             = (intersection + epsilon) / (union + epsilon)

    print("iou:", iou)
        
    return iou


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_box_iou(bb, BBGT):

    ovmax = -np.inf

    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    #jmax = np.argmax(overlaps)

    return ovmax


def calc_ap(gt_img, pred_img, gt_bbox, bboxs):
    
    #predh, predw, predc = pred_img.shape
    #gth, gtw, gtc = gt_img.shape

    #h_ratio = gth/predh
    #w_ratio = gtw/predw

    BBGT = np.array(gt_bbox, dtype=np.float32).reshape(-1, 4)

    blen = len(bboxs)
    if blen == 0:
        return 0.

    tp = np.zeros(blen)
    fp = np.zeros(blen)
    for i in range(blen):
        #bb  = np.array(bboxs[i][0]*w_ratio, bboxs[i][1]*h_ratio,bboxs[i][2]*w_ratio, bboxs[i][3]*h_ratio], np.float32)
        bb  = np.array(bboxs[i], np.float32)
        iou = calc_box_iou(bb, BBGT)
        if iou >= 0.5:
            tp[i] = 1.
        else:
            fp[i] = 1

    recall = sum(tp)/(float(len(BBGT)) + 1e-6)
    precision = sum(tp)/(float(blen) + 1e-6)
    #print(recall, precision)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(len(BBGT)) + 1e-6)

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=False)

    return ap


def calc_cate_ap(cate_bb):

    BBGT = []

    bbox_idx = {}
    gtbbox_idx = {}
    BBGTS = []
    i = 0
    for img_id in cate_bb:
        bboxs = cate_bb[img_id]['predbbox']
        gt_bbox = cate_bb[img_id]['gtbox']
        BBGTS += gt_bbox
        for bbox in bboxs:
            bbox_idx[i] = np.array(bbox, np.float32)
            gtbbox_idx[i] = np.array(gt_bbox, dtype=np.float32).reshape(-1, 4)
            i += 1


    if i == 0:
        return 0.

    blen = i

    tp = np.zeros(blen)
    fp = np.zeros(blen)
    for x in range(blen):
        BBGT = gtbbox_idx[x]
        #bb  = np.array(bboxs[i][0]*w_ratio, bboxs[i][1]*h_ratio,bboxs[i][2]*w_ratio, bboxs[i][3]*h_ratio], np.float32)
        bb  = np.array(bbox_idx[x], np.float32)
        iou = calc_box_iou(bb, BBGT)
        if iou >= 0.5:
            tp[x] = 1.
        else:
            fp[x] = 1

    recall = sum(tp)/(float(len(BBGTS)) + 1e-6)
    precision = sum(tp)/(float(blen) + 1e-6)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(len(BBGTS)) + 1e-6)

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=True)

    return ap


def calc_miou():
    
    gt_img = cv2.imread(os.path.join(test_path, det_p, det_p+'_1.jpg'))  # groundtruth

    pred_img = cv2.imread(os.path.join(test_path, det_p, det_p+'_1.jpg'))
    iou = calc_iou(gt_img, pred_img)
    cls_iou[img_id][cls] = iou
    
    ious = []
    for img_id in cls_iou:
        ious.append(np.mean(list(cls_iou[img_id].values())))


    print('the mIoU is {}'.format(np.mean(ious)))


def generate_pets_gt(oxford_pets_root, save_root, tasks):
    
    print("Begin to generate oxford-pets ground truth.")

    for line in open(os.path.join(oxford_pets_root, 'annotations/det_test.txt')):
        
        line                        = line.strip()
        words                       = line.split(' ')
        img_id                      = words[0].split(".")[0] #'Abyssinian_201'

        target_name = ' '.join(img_id.split('_')[:-1]).strip()
        
        for task_type in tasks:
            
            if task_type == 'seg':
                
                output_img          = get_seg_img(oxford_pets_root, img_id)
                
                output_path         = os.path.join(save_root, img_id + "_" + task_type)
                
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)

                output_img.save(output_path+'/{}_{}_gt.jpg'.format(img_id, task_type))

            else:
                output_img, bbox    = get_bbox_img(oxford_pets_root, img_id, None, dataset='oxford-pets')
                
                if output_img is None:
                    continue
                
                output_path         = os.path.join(save_root, img_id + "_" + task_type) # ./data/image_pairs_evalation/Abyssinian_201_det/
                
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)
                
                bbox_info = {'bbox': bbox}
                bbox_file = open(os.path.join(output_path, 'bbox.json'), 'w')
                bbox_file.write(json.dumps(bbox_info))
                bbox_file.close()
                output_img.save(output_path+'/{}_{}_gt.jpg'.format(img_id, task_type)) # ./data/image_pairs_evalation/Abyssinian_201_det/Abyssinian_201_det_gt.jpg

    return



class genGT(object):
    
    def __init__(self, dataset_root, save_root, task, split=None, test_txt_path=None):
        
        for i in range(len(CLASSES)):
            self.cls_ade_dict[i] = CLASSES[i]
        
        self.dataset_root   = dataset_root
        self.save_root      = save_root
        self.split          = split
        self.task           = task
        self.test_txt_path  = test_txt_path
        
    def generate_ade20k_gt(self):
        
        print('Begin to generate ADE20k ground truth')
        
        for img_name in open(os.path.join(self.dataset_root, self.split)):
        
            img_name = img_name.strip()
            
            img_path                     = os.path.join(self.dataset_root, "images/validation", img_name)
            seg_path                     = os.path.join(self.dataset_root, "annotations/validation", img_name.split(".")[0]+".png")
            anno                         = Image.open(seg_path)
            anno                         = np.array(anno)
            
            clses = np.unique(anno)
            
            for cls in clses: # e.g., cls=1
                
                img                      = Image.open(img_path) #original image
                seg_img                  = Image.new('RGB',(img.size[0],img.size[1]))
                seg_img                  = np.array(seg_img)

                #find where equals cls in anno
                r, c                     = np.where(anno == cls) #r,c are arraries
                
                for i in range(len(r)):
                    
                    seg_img[r[i],c[i],:] = (255,255,255)

                seg_img = Image.fromarray(seg_img)
                
                cls_name = self.cls_ade_dict[cls]
                
                if cls_name == "background":
                    continue
                
                out_name                 = img_name+ "_" + cls_name + self.task
                output_path              = os.path.join(self.save_root, out_name)
                
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)

                seg_img.save(output_path + '/{}_gt.jpg'.format(out_name))
        
        return

    def generate_nyuv2_gt(self):
        
        print('Begin to generate NYU_V2 ground truth')
        
        with open(self.test_txt_path) as file:  

            for line in file:
                
                img_path_part   = line.strip().split(" ")[0] # kitchen/rgb_00045.jpg
                file_name       = img_path_part.split("/")[0] # kitchen
                img_name        = img_path_part.split("/")[1] # rgb_00045.jpg
                img_id          = file_name + "_" + img_name.split(".")[0] # kitchen_rgb_00045
                
                dep_path_part   = file_name + "/vli_depth_" + img_name.split("_")[-1].replace("jpg","png") # kitchen_0028b/vli_depth_00045.jpg
                dep_path        = os.path.join(self.dataset_root, dep_path_part)
                
                depth_img       = Image.open(dep_path).convert("RGB")
                
                output_path     = os.path.join(self.save_root, img_id + "_" + self.task)
                    
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)
                        
                depth_img.save(output_path+'/{}_{}_gt.jpg'.format(img_id, self.task))
        
        return

    def generate_coco_gt(self):
        '''
        Generate coco gt bbox.json and images with g.t. bboxes.
        '''
        
        print('Begin to generate COCO ground truth: box.json and g.t img')
        
        img_info, clses       = preproc_coco(self.dataset_root)
        
        for img_name in open(os.path.join(self.dataset_root,self.split)):
        
            img_name              = img_name.strip()
            image_id              = img_name.split(".")[0] #000001234
            id                    = image_id.lstrip("0") #1234
            
            for cid in img_info[id]:
                
                cname = clses[cid] #target_name  
                bbox  = img_info[id][cid]['bbox']
                
                # save det image
                output_path = os.path.join(self.save_root, image_id + '_{}_det'.format(cname))
                det_img, bbox = get_bbox_img(self.dataset_root, image_id, bbox, dataset='MSCOCO')
                det_img.save(os.path.join(output_path, "{}_{}_det_gt.jpg".format(image_id, cname)))
                
                # save g.t box.json
                bbox_info = {'bbox': bbox}
                bbox_file = open(os.path.join(output_path, 'bbox.json'), 'w')
                bbox_file.write(json.dumps(bbox_info))
                bbox_file.close()
           
        return
    
def get_color(color_dict, image_path):
    
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None

    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        img, cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in img:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
    return color


def getColorList():
    '''
    Create color dict
    '''
    
    color_dict          = {}
    
    # black
    lower_black         = np.array([0, 0, 0])
    upper_black         = np.array([180, 255, 46])
    color_list          = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    color_dict['black'] = color_list

    # grey
    lower_gray          = np.array([0, 0, 46])
    upper_gray          = np.array([180, 43, 220])
    color_list          = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    color_dict['grey'] = color_list

    # white
    lower_white         = np.array([0, 0, 221])
    upper_white         = np.array([180, 30, 255])
    color_list          = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    color_dict['white'] = color_list

    # red
    lower_red           = np.array([156, 43, 46])
    upper_red           = np.array([180, 255, 255])
    color_list          = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    color_dict['red'] = color_list

    # red2
    lower_red           = np.array([0, 43, 46])
    upper_red           = np.array([10, 255, 255])
    color_list          = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    color_dict['red2']  = color_list

    # orange
    lower_orange        = np.array([11, 43, 46])
    upper_orange        = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    color_dict['orange'] = color_list

    # yellow
    lower_yellow        = np.array([26, 43, 46])
    upper_yellow        = np.array([34, 255, 255])
    color_list          = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    color_dict['yellow'] = color_list

    # green
    lower_green         = np.array([35, 43, 46])
    upper_green         = np.array([77, 255, 255])
    color_list          = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    color_dict['green'] = color_list

    # cyan
    lower_cyan          = np.array([78, 43, 46])
    upper_cyan          = np.array([99, 255, 255])
    color_list          = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    color_dict['cyan']  = color_list

    # blue
    lower_blue          = np.array([100, 43, 46])
    upper_blue          = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    color_dict['blue'] = color_list

    # purple
    lower_purple        = np.array([125, 43, 46])
    upper_purple        = np.array([155, 255, 255])
    color_list          = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    color_dict['purple'] = color_list
 
    return color_dict
    
    
def evaluate_cls(cls_pred_root):
    
    count, true             = 0,0
    # create color_dict
    color_dict              = getColorList()           

    img_list = os.listdir(cls_pred_root)
    for img_name in img_list:
        img_path            = os.path.join(cls_pred_root, img_name)
        # get color name
        color               = get_color(color_dict, img_path)
        
        # calc acc
        if color == "blue":
            
            true           += 1
            
        count              += 1
    
    acc                     =true / count
    
    print("acc:", acc)
        
    
    
    
    return




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--oxford_pets_root", default='./data/oxford-pets', type=str)
    parser.add_argument("--nyuv2_root", default='./data/nyu_mdet', type=str)
    parser.add_argument("--gt_path", default='./data/image_pairs_pets_nyuv2', type=str)
    parser.add_argument("--pred_path", default='./imgs_test_oxford_pets', type=str)
    parser.add_argument("--cls_pred_root", default='./outputs/imgs_test_oxford_pets', type=str)
    parser.add_argument("--coco_root", default='./data/coco', type=str)
    parser.add_argument("--ade20k_root", default='./data/ADEChallengeData2016', type=str)
    parser.add_argument("--save_root", default='./data/image_pairs_evaluation_dep', type=str)
    parser.add_argument("--save_ade_root", default='./outputs/imgs_test_ade20k', type=str)
    parser.add_argument("--tasks", default=['seg', 'det'], nargs='+')
    
    args                             = parser.parse_args()
    cls_iou, cls_ade_dict, cls_ap    = {}, {},{}
    n                                = 0
    
    for i in range(len(CLASSES)):
        cls_ade_dict[i] = CLASSES[i]
    
    #TODO merge generating gt into edit_cli.py
    
    # generate_ade20k_gt(args.ade20k_root, args.save_ade_root, cls_ade_dict)
    
    # generate_nyuv2_gt(args.nyuv2_root, args.save_root)
    
    # generate_sunrgbd_gt(args.sunrgbd_root, args.save_root)
    
    # generate_pets_gt(args.oxford_pets_root, args.save_root, args.tasks)
    
    # generate_coco_gt(args.coco_root, args.save_root)
    
    # calc acc
    # acc = evaluate_cls(args.cls_pred_root)
    
    test_path = './outputs/imgs_test_fs1000'
    cls_iou = {}
    cls_ap = {}
    cate_bb = {}
    n = 0
    
    for det_p in os.listdir(test_path): #ADE_train_00000001.jpg_ashcan_seg
        if det_p in ['.DS_Store', 'seeds.json']:
            continue

        pinfo  = det_p.split('_')
        img_id = pinfo[2] # Abyssinian
        task_type = pinfo[-1] #seg
        cls  = pinfo[-2] #1,2,3,...

        if img_id not in cls_iou:
            cls_iou[img_id] = {}
        
        if img_id not in cls_ap:
            cls_ap[img_id] = {}
            
        if cls not in cate_bb:
            cate_bb[cls] = {}

        gt_img_root = os.path.join(test_path, det_p, det_p+'_gt.jpg')
        # gt_img = cv2.imread(os.path.join(test_path, det_p, det_p+'_gt.jpg'))  # groundtruth
        # gt_img = Image.open(os.path.join(test_path, det_p, det_p+'_gt.jpg'))  # groundtruth
        if task_type == 'seg':
            pred_path = os.path.join(test_path, det_p, det_p+'_pred.jpg')
            if not os.path.exists(pred_path):
                continue
            
            pred_img  = Image.open(pred_path)
            iou = calc_iou(gt_img_root, pred_img)
            cls_iou[img_id][cls] = iou

        elif task_type == 'det': # 检测
            
            pred_img                = cv2.imread(os.path.join(test_path, det_p, det_p+'_pred.jpg'))
            gt_img                  = cv2.imread(os.path.join(test_path, det_p, det_p+'_gt.jpg'))  # groundtruth

            h, w, c                 = gt_img.shape
            
            pred_img                = cv2.resize(pred_img, (w, h))

            box_fp                  = open(os.path.join(test_path, det_p, 'bbox.json'))
            box_pr                  = open(os.path.join(test_path, det_p, 'pred_bbox.json'))
            gt_bbox                 = json.loads(box_fp.readline())['bbox']
            
            bboxs                   = json.loads(box_pr.readline())['pred_bbox']
            pred_bboxes             = cal_bboxes_iou(gt_bbox, bboxs)
            ap                      = calc_ap(gt_img, pred_img, gt_bbox, pred_bboxes)
            cls_ap[img_id][cls]     = ap
            cate_bb[cls][img_id]    = {'predbbox': bboxs, 'gtbox': gt_bbox}

        else:
            continue
        
        n += 1
        if n % 100 == 0:
            print('{} test sample processed!'.format(n))
            #break

    ious = []
    for img_id in cls_iou:
        iou_ = np.mean(list(cls_iou[img_id].values()))
        if math.isnan(iou_):
            continue 
        ious.append(iou_)
    


    # print('the mIoU is {}'.format(np.mean(ious)))


    # APs = []
    # for img_id in cls_iou:
    #     if len(list(cls_ap[img_id].values())) == 0:
    #         continue
    #     APs.append(np.mean(list(cls_ap[img_id].values())))

    # print('the mAP is {}'.format(np.mean(APs)))
    
    # cAPs = []
    # for cls in cate_bb:
    #     if len(cate_bb[cls]) == 0:
    #         continue
    #     ap = calc_cate_ap(cate_bb[cls])
    #     cAPs.append(ap)

    # print('the mAP of class is {}'.format(np.mean(cAPs)))
        