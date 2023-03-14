import numpy as np
import cv2
import os
import pdb
from PIL import Image, ImageChops
from torchvision import transforms

ori_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/imgs/000000000063_test.jpg'
pre_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/imgs_test/000000000063.jpg'
save_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/imgs/0000000000063_1234'

def ShapeDetection(img):
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #转灰度图
    #imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  #高斯模糊

    #binary = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #ret, binary = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, binary = cv2.threshold(imgGray, 10, 255, 0)
    #binary = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    kernel = np.ones((5,5),np.uint8) 
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #binary = cv2.erode(binary, kernel, iterations = 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    imgCanny = cv2.Canny(binary,60,60)  #Canny算子边缘检测

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #寻找轮廓点

    bboxs = []
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        #cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长

        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        if CornerNum != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

        bbox = (x,y, x+w,y+h)
        bboxs.append(bbox)

        cv2.rectangle(imgContour, (x,y), (x+w,y+h), (0,0,255), 2)  #绘制边界框

    return imgContour, bboxs


resize = transforms.Resize([512,512])

ori_img = Image.open(ori_path)
ori_img = resize(ori_img)

det_img = Image.open(pre_path)
pdb.set_trace()
box_img = ImageChops.difference(det_img, ori_img)

# imgContour, bbox = ShapeDetection(box_img)

# cv2.imwrite((save_path+'_box.jpg'), imgContour)
box_img.save(save_path+'_box.jpg')
