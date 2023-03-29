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
        
        if area != 0 and perimeter != 0 and area / perimeter > 0.95:  # remove non-square
            conut = 0
            for i in np.arange(x, x + w, 1):
                for j in np.arange(y, y + h, 1):
                    if binary[j, i] != 0:
                        conut = conut + 1

            if conut > 0.8 * w * h: # remove noise in square
                for i in np.arange(x, x + w, 1):
                    for j in np.arange(y, y + h, 1):
                        binary[j, i] = 0

                        bbox[0] = x
                        bbox[1] = y
                        bbox[2] = x+w
                        bbox[3] = y+h
                        bbox[4] = 1 #confidence =1 when we are sure there has a rectangle
                        bboxes.append(list(bbox))
        
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
    

def generate_exc_bbox(pred_root):
    '''
    generate extracted bboxes and visual images. (pred_bbox.json & images)
    '''
    
    file_list                           = os.listdir(pred_root)
    n                                   = 0
    
    for file in file_list:
        
        file_path                       = os.path.join(pred_root, file)
        
        img_list                        = os.listdir(file_path)
        
        for img in img_list:
            
            # # resume
            # if args.resume:
            #     file_                       = file + "+exc.jpg"
            #     print("pass,{}".format(file_))
            #     if file_ in img_list:
            #         continue
            
            
            if fnmatch(img, "*pred.jpg"):
                pred_img_path           = os.path.join(file_path, img)

            else:
                continue
            
            # generate coordinates of bboxes (json)
            gt_path                     = os.path.join(root, 'val2017', img.split("_")[0]+'.jpg')
            bboxes, output              = extract_bbox(pred_img_path, gt_path)
            bbox_info                   = {'pred_bbox': bboxes}
            bbox_file                   = open(os.path.join(file_path, 'pred_bbox.json'), 'w')
            bbox_file.write(json.dumps(bbox_info))
            bbox_file.close()
            
            save_path                   = os.path.join(file_path, img.replace("pred", "exc"))
            
            # generate images with bboxes
            cv2.imwrite(save_path, output)
            
            n+=1
            if n % 100 == 0:
                print("{} images done".format(n))
    
    return


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--resume", default=0, type=int)
    
    args                             = parser.parse_args()
    
    root                                = "./data/coco"
    pred_root                           = "./outputs/imgs_test_coco"

    generate_exc_bbox(pred_root)
    
    