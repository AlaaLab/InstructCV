import cv2
import numpy as np


img = cv2.imread('/lustre/grp/gyqlab/lism/brt/language-vision-interface/image_pairs/000000000632_det_potted plant/000000000632_det_potted plant_1.jpg')

# 转换成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 数据类型转换成float32
gray_float32 = np.float32(gray)

# 角点检测
dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)

#设置阈值,将角点绘制出来,阈值根据图像进行选择
R=dst.max() * 0.01
#这里将阈值设为dst.max()*0.01 只有大于这个值的数才认为数角点
img[dst > R] = [0, 0, 255]

cv2.imwrite("aaaa.jpg", img)


