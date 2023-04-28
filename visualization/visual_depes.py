import cv2
import numpy as np


img_gray = cv2.imread('/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/img_test_cls/demo/demo_depes_pred.jpg', 0)
n_min    = np.min(img_gray)
n_max    = np.max(img_gray)
img_gray = (img_gray-n_min)/(n_max-n_min+1e-8)
img_gray = (255*img_gray).astype(np.uint8)
# import pdb;pdb.set_trace()
img_pseudo = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
cv2.imwrite("/lustre/grp/gyqlab/lism/brt/language-vision-interface/outputs/img_test_cls/demo/demo_depes_vis.jpg", img_pseudo)