import cv2
from argparse import ArgumentParser
import numpy as np
import os
import pdb
import json
from fnmatch import fnmatch
import time


def cnt_area(cnt):
    '''
    sorting contours by contour area size
    '''
    area = cv2.contourArea(cnt)
    
    return area


def extract_bbox(pred_img_path, gt_path):
    '''
    input one image
    return coodinates of all bboxes in this image
    '''
    start                       = time.time()
    gt_img                      = cv2.imread(gt_path)
    h, w, c                     = gt_img.shape  
    bgr_img                     = cv2.imread(pred_img_path)     # load images
    bgr_img                     = cv2.resize(bgr_img, (w, h))
    output                      = bgr_img
    bgr_img                     = cv2.medianBlur(bgr_img, 3)    # median filtering
    bgr_img                     = cv2.bilateralFilter(bgr_img, 0, 0, 30)   # bilateral filtering
    output_img                  = bgr_img

    # extracting red areas using hsv channels
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 120, 66])
    high_hsv = np.array([8, 255, 255])
    mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    low_hsv = np.array([165, 120, 66])  # 46
    high_hsv = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    # extracting the red area of the original image
    for i in np.arange(0, bgr_img.shape[0], 1):
        for j in np.arange(0, bgr_img.shape[1], 1):
            if mask1[i, j] == 0 and mask2[i, j] == 0:
                output_img[i, j, :] = 0

    # get grayscale map
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # threshold segmentation based on grayscale
    otsuThe, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst_Otsu = cv2.Canny(dst_Otsu, 50, 150, apertureSize=3)

    # image binarization
    ret, binary = cv2.threshold(dst_Otsu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    # find the outline
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours = list(contours)
    contours.sort(key=cnt_area, reverse=False)

    bboxes = []

    # contour judgment
    for obj in contours: # objs in one image

        bbox = []

        area = cv2.contourArea(obj)  # calculate the area of the area within the contour
        perimeter = cv2.arcLength(obj, True)  # calculate the contour perimeter
        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # get the coordinates of the contour corner points
        x, y, w, h = cv2.boundingRect(approx)  # get coordinate values and width and height

        if perimeter < 20:  # remove small contour areas
            for i in np.arange(x, x + w, 1):
                for j in np.arange(y, y + h, 1):
                    binary[j, i] = 0
        
        else:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 1)  # draw the bounding box
            bbox = np.zeros((4,)) # print border coordinates
            
            bbox[0] = x
            bbox[1] = y
            bbox[2] = x+w
            bbox[3] = y+h
            bboxes.append(list(bbox))
        
    end             = time.time()
    
    print("one image done, cost time:{}".format(end-start))
    
    return bboxes, output
  
def extract_bbox_filter(pred_img_path, gt_path):
    '''
    input one image
    return coodinates of all bboxes in this image
    '''
    start                       = time.time()
    gt_img                      = cv2.imread(gt_path)
    h, w, c                     = gt_img.shape  
    bgr_img                     = cv2.imread(pred_img_path)     # load images
    bgr_img                     = cv2.resize(bgr_img, (w, h))
    output                      = bgr_img
    bgr_img                     = cv2.medianBlur(bgr_img, 3)    # median filtering
    bgr_img                     = cv2.bilateralFilter(bgr_img, 0, 0, 30)   # bilateral filtering
    output_img                  = bgr_img

    # extracting red areas using hsv channels
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 120, 66])
    high_hsv = np.array([8, 255, 255])
    mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    low_hsv = np.array([165, 120, 66])  # 46
    high_hsv = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    # extracting the red area of the original image
    for i in np.arange(0, bgr_img.shape[0], 1):
        for j in np.arange(0, bgr_img.shape[1], 1):
            if mask1[i, j] == 0 and mask2[i, j] == 0:
                output_img[i, j, :] = 0

    # get grayscale map
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # threshold segmentation based on grayscale
    otsuThe, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst_Otsu = cv2.Canny(dst_Otsu, 50, 150, apertureSize=3)

    # image binarization
    ret, binary = cv2.threshold(dst_Otsu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    # find the outline
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours = list(contours)
    contours.sort(key=cnt_area, reverse=False)

    bboxes = []
    perimeter_ = []
    # contour judgment
    for obj in contours:

        area = cv2.contourArea(obj)
        perimeter = cv2.arcLength(obj, True)
        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        if perimeter > 700 or perimeter < 50 or 601>perimeter>450:
            
            for i in np.arange(x, x + w, 1):
                for j in np.arange(y, y + h, 1):
                    binary[j, i] = 0
        else:           
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 绘制边界框
        # elif area != 0 and perimeter != 0:
            
        #     conut = 0
        #     for i in np.arange(x, x + w, 1):
        #         for j in np.arange(y, y + h, 1):
        #             if binary[j, i] != 0:
        #                 conut = conut + 1
        #     # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 绘制边界框
        #     if conut > 0.8 * w * h:
        #         for i in np.arange(x, x + w, 1):
        #             for j in np.arange(y, y + h, 1):
        #                 binary[j, i] = 0
        #     # else:
        #     #     cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 绘制边界框

            
        # else:
        #     for i in np.arange(x, x + w, 1):
        #         for j in np.arange(y, y + h, 1):
        #             binary[j, i] = 0
            
        
    return bboxes, output
  

def vis_exc_bbox(pred_root, gt_path, save_path):
    '''
    visualize extracted bboxes and visual images. (pred_bbox.json & images)
    '''
    
    
    bboxes, output              = extract_bbox_filter(pred_root, gt_path)
    
    # generate images with bboxes
    cv2.imwrite(save_path, output)


    return


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--resume", default=0, type=int)
    
    args                             = parser.parse_args()
    
    root                                = "./data/coco"
    pred_root                           = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/imgs_test_coco/000000219271_truck_det/000000219271_truck_det_pred.jpg"
    save_path                           = "./outputs/imgs_test_coco_save/123_.jpg"
    gt_path                             = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/coco/val2017/000000219271.jpg"

    # vis_exc_bbox(pred_root, gt_path, save_path)
    pred_bbox = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/imgs_test_coco/000000219271_truck_det/bbox.json'
    box_pr                  = open(pred_bbox)
    gt_bbox                 = json.loads(box_pr.readline())['bbox']
    output = cv2.imread(pred_root)
    gt     = cv2.imread(gt_path)
    h,w,c  = gt.shape
    output = cv2.resize(output,(w,h))
    print(type(gt_bbox))
    
    