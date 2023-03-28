import numpy as np
import cv2
import os
import pdb
from PIL import Image, ImageChops
from torchvision import transforms

ori_path = './outputs/img_test_det_car/000000009891_det_car_0.jpg'
pre_path = './outputs/imgs_coco_output2/000000009891_det_car_0_test.jpg'
save_path = './outputs/imgs_coco_output2/000000009891_test_sub'

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


def image_diff(image1, image2, threshold=0):
    """Detect the differences between two images and ignore small differences.
    
    Args:
        image1 (numpy.ndarray): The first image as a NumPy array.
        image2 (numpy.ndarray): The second image as a NumPy array.
        threshold (int): The threshold for the pixel intensity difference between the two images. Defaults to 30.
    
    Returns:
        numpy.ndarray: The difference image as a NumPy array.
    """
    
    height, width, _ = image1.shape
    image2 = cv2.resize(image2, (width, height))
    
    # Calculate the absolute difference between the two images
    diff = cv2.absdiff(image1, image2)
    
    # Apply threshold to ignore small differences
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Apply a morphological operation to remove small regions of noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the difference image
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    
    return diff


# det_img         = Image.open(pre_path)
# h,w           = det_img.size
# det_img             = np.array(det_img)

# resize              = transforms.Resize([w,h])

# ori_img             = Image.open(ori_path)
# ori_img             = resize(ori_img)
# ori_img             = np.array(ori_img)
# det_img             = np.array(det_img)

det_img         = cv2.imread(pre_path)
ori_img         = cv2.imread(ori_path)
# pdb.set_trace()

# box_img         = ImageChops.difference(det_img, ori_img)
box_img             = image_diff(det_img, ori_img)

# imgContour, bbox = ShapeDetection(box_img)
# box_img             = Image.fromarray(box_img)
cv2.imwrite(save_path+'_box.jpg', box_img)

hsv = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(save_path+'_box333.jpg', hsv)







