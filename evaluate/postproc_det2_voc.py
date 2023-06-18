# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Post-steps for evaluating object detection task.
# Generate the extracted bboxes and visualization images. (pred_bbox.json & images)
# --------------------------------------------------------

import cv2
from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
import pdb
import json
from fnmatch import fnmatch
import time
import shutil

class postDet_voc(object):
    
    def __init__(self, root = './data/VOCdevkit/VOC2012',
                 pred_root='./outputs/imgs_test_voc_rp', vis=True):
        
        self.root           = root
        self.pred_root      = pred_root
        self.vis            = vis

    def cnt_area(self, cnt):
        '''
        sorting contours by contour area size
        '''
        area = cv2.contourArea(cnt)
        
        return area

    def extract_bbox(self, pred_img_path, gt_path):
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

        # extracting blue areas using hsv channels
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # low_hsv = np.array([0, 120, 66])
        # high_hsv = np.array([8, 255, 255])
        low_hsv = np.array([100,130,50])
        high_hsv = np.array([125,255,255])
        mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

        # low_hsv = np.array([165, 120, 66])  # 46
        # high_hsv = np.array([180, 255, 255])
        # low_hsv = np.array([100,43,46])
        # high_hsv = np.array([140,255,255])
        # mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

        # extracting the red area of the original image
        for i in np.arange(0, bgr_img.shape[0], 1):
            for j in np.arange(0, bgr_img.shape[1], 1):
                # if mask1[i, j] == 0 and mask2[i, j] == 0:
                if mask1[i, j] == 0:
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
        contours.sort(key=self.cnt_area, reverse=False)

        bboxes = []

        # contour judgment
        for obj in contours: # objs in one image

            bbox = []

            area = cv2.contourArea(obj)  # calculate the area of the area within the contour
            perimeter = cv2.arcLength(obj, True)  # calculate the contour perimeter
            approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # get the coordinates of the contour corner points
            x, y, w, h = cv2.boundingRect(approx)  # get coordinate values and width and height

            if perimeter < 10:  # remove small contour areas
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
    
    def delete(self, points):
        points = np.asarray(points)
        goal = []
        for i in range(points.shape[0]-1):
            a = abs(points[0][0] - points[i+1][0])
            b = abs(points[0][1] - points[i+1][1])
            if a > 5 or b > 5:
                goal.append(points[i+1])
        goal.append(points[0])
        if len(goal) != points.shape[0]:
            goal = self.delete(goal)
        return goal

    def ext_coor(self, img_path, gt_path):

        print("img_path", img_path)
        img_name = img_path.split('_')[-2]
        img = cv2.imread(img_path)
        gt_img                      = cv2.imread(gt_path)
        h, w, c                     = gt_img.shape  
        img                         = cv2.resize(img, (w, h))
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        low_hsv = np.array([100,110,70])
        high_hsv = np.array([140,255,255])
        # low_hsv = np.array([100,43,46])
        # high_hsv = np.array([140,255,255])
        mask1 = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
        # low_hsv = np.array([156,90,90])
        # high_hsv = np.array([180,255,255])
        # mask2 = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
        # mask = cv2.add(mask2,mask1)
        kernel = np.ones((3,3),'uint8')
        mask = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel,iterations=1)
        mask = cv2.copyMakeBorder(mask,50,50,50,50,cv2.BORDER_CONSTANT,value=0)

        points = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.sum(np.array(mask[i+50,j+50:j+70]) )== 255*20 and np.sum(np.array(mask[i+50:i+70,j+50]) )== 255*20 \
                        and np.sum(np.array(mask[i+50:i+60,j+40:j+50]) )< 255*50 \
                        and np.sum(np.array(mask[i+40:i+50,j+50:j+60]) )< 255*50\
                        and np.sum(np.array(mask[i+40:i+50,j+40:j+50]) )< 255*50\
                        and np.sum(np.array(mask[i+50:i+60,j+50:j+60]) )< 255*50:

                    cv2.circle(img,(j,i),1,(0,255,0),-1)
                    points.append([i, j])
                if np.sum(np.array(mask[i+50,j+30:j+50]) )== 255*20 and np.sum(np.array(mask[i+30:i+50,j+50]) )== 255*20\
                        and np.sum(np.array(mask[i+40:i+50,j+50:j+60]) )< 255*50\
                        and np.sum(np.array(mask[i+50:i+60,j+40:j+50]) )< 255*50\
                        and np.sum(np.array(mask[i+50:i+60,j+50:j+60]) )< 255*50\
                        and np.sum(np.array(mask[i+40:i+50,j+40:j+50]) )< 255*50:
                        cv2.circle(img,(j,i),2,(0,255,0),-1)
                        points.append([i, j])

        points = np.array(points)
        if points.size == 0:
            return [], img
        
        points_sort = pd.DataFrame(points,columns=['x','y'])
        points_sort.sort_values(by=['x','y'],axis=0)


        goal = self.delete(points)
        goal = pd.DataFrame(goal,columns=['x','y'])
        goal = goal.sort_values(by=['x','y'],axis=0)
        goal = np.array(goal)
        point = []
        for i in range(goal.shape[0]):
            for j in np.arange(i+1,goal.shape[0]):
                point.append([goal[i,0],goal[i,1],goal[j,0],goal[j,1]])
        point_new = []
        for i in range(len(point)):
            if point[i][1] < point[i][3]:
                point_new.append(point[i])
        
        img_vis = img.copy()
        
        if len(point_new) == 0:
            return [], img
        
        bboxes = []
        for i in range(len(point_new)):
            xx1 = point_new[i][1]
            yy1 = point_new[i][0]
            xx2 = point_new[i][3]
            yy2 = point_new[i][2]
            cv2.rectangle(img_vis, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)
            
            bbox = [int(xx1),int(yy1),int(xx2),int(yy2)]
            bboxes.append(bbox)
            cv2.putText(img_vis, img_name, (xx1 + 10, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        return bboxes, img_vis
    
    def generate_exc_bbox(self):
        '''
        generate extracted bboxes and visualization images. (pred_bbox.json & images)
        args: pred_root  - path where stores the pred results & path to save the pred_bbox.json and images
            vis        - bool, if generate visulization images (yellow bboxes) or not
        return: none
        '''
        
        file_list                           = os.listdir(self.pred_root)
        file_list                           = sorted(file_list)
        print("len(file_list)", len(file_list)) #12808
        n                                   = 0

        for file in file_list[0:len(file_list)]:
            
            file_path                       = os.path.join(self.pred_root, file)
            img_list                        = os.listdir(file_path)
            
            for img in img_list:
                
                # resume
                file_                       = file + "_exc.jpg"
                if file_ in img_list:
                    print("pass,{}".format(file_))
                    continue
                
                
                if fnmatch(img, "*pred.jpg"):
                    pred_img_path           = os.path.join(file_path, img)

                else:
                    continue

                # generate coordinates of bboxes (json)
                gt_path                     = os.path.join(self.root, 'JPEGImages', img.split("_")[0] + "_" + img.split("_")[1]+'.jpg')
                bboxes, output              = self.extract_bbox(pred_img_path, gt_path)
                # diff, img_id              = self.image_diff(pred_img_path) #do subtract
                
                #for debugging: save diff image
                # cv2.imwrite(os.path.join(file_path, '{}_diff.jpg'.format(img_id)), diff)
                
                bboxes2, img_vis            = self.ext_coor(pred_img_path, gt_path)
                bboxes = bboxes + bboxes2
                
                # bbox_info                   = {'pred_bbox': bboxes}
                bbox_info                   = {'pred_bbox': bboxes}
                bbox_file                   = open(os.path.join(file_path, 'pred_bbox.json'), 'w')
                bbox_file.write(json.dumps(bbox_info))
                bbox_file.close()
                
                save_path                   = os.path.join(file_path, img.replace("pred", "exc"))
                save_path2                  = os.path.join(file_path, img.replace("pred", "exc2"))
                
                # generate images with bboxes
                if self.vis:
                    cv2.imwrite(save_path, output)
                    cv2.imwrite(save_path2, img_vis)
                
                n+=1
                if n % 100 == 0:
                    print("{} images done".format(n))
        
        return


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--resume", default=0, type=int)
    
    args                             = parser.parse_args()
    
    # root                                = "./data/coco"
    # pred_root                           = "./outputs/imgs_test_coco"

    postDet().generate_exc_bbox()
    
    