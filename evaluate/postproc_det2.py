import cv2
import numpy as np


path = "./outputs/imgs_coco_output3/000000001532_det_car_0_test.jpg"

# 读取彩色图像
bgr_img = cv2.imread(path)
output = bgr_img
# 中值滤波
bgr_img = cv2.medianBlur(bgr_img, 3)
# 双边滤波
bgr_img = cv2.bilateralFilter(bgr_img, 0, 0, 30)
# 输出图像
output_img = bgr_img
# 利用hsv通道提取红色区域
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
# low_hsv = np.array([0, 120, 66])
# high_hsv = np.array([8, 255, 255])
low_hsv = np.array([0, 127, 128])
high_hsv = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

# low_hsv = np.array([165, 120, 66])  # 46
# high_hsv = np.array([180, 255, 255])
low_hsv = np.array([165, 120, 66])  # 46
high_hsv = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

# 提取原始图像红色区域
for i in np.arange(0, bgr_img.shape[0], 1):
    for j in np.arange(0, bgr_img.shape[1], 1):
        if mask1[i, j] == 0 and mask2[i, j] == 0:
            output_img[i, j, :] = 0


# 得到灰度图
gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
# 基于灰度进行阈值分割
otsuThe, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# 图像二值化
ret, binary = cv2.threshold(dst_Otsu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

# 寻找轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# 对轮廓按照轮廓面积大小进行排序
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


contours = list(contours)
contours.sort(key=cnt_area, reverse=False)

# 轮廓判断
for obj in contours:

    area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
    perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
    approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
    x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度

    if area < 100:  # 去除小面积轮廓
        for i in np.arange(x, x + w, 1):
            for j in np.arange(y, y + h, 1):
                binary[j, i] = 0
    elif area != 0 and perimeter != 0 and area / perimeter > 0.95:  # 去除非矩形轮廓
        conut = 0
        for i in np.arange(x, x + w, 1):
            for j in np.arange(y, y + h, 1):
                if binary[j, i] != 0:
                    conut = conut + 1

        if conut > 0.8 * w * h: # 去除轮廓内存在大量干扰像素的轮廓
            for i in np.arange(x, x + w, 1):
                for j in np.arange(y, y + h, 1):
                    binary[j, i] = 0
        else:
            # 输出
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 绘制边界框
            point1 = np.zeros((2,2)) # 打印边框坐标
            point1[0,0] = x
            point1[0,1] = x+w
            point1[1,0] = y
            point1[1,1] = y+h
            print(point1)
    else:
        for i in np.arange(x, x + w, 1):
            for j in np.arange(y, y + h, 1):
                binary[j, i] = 0
# 输出的图像
cv2.imwrite("1234567.png",output)
