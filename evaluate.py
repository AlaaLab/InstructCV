import numpy as np
from PIL import Image
import pdb


def read_gt_img(img_name):
    
    img = Image.open(img_name)
    np_img = np.array(img)
    np_img = np.asarray([np_img], dtype=np.int32)
    
    return np_img

def read_pred_img(img_name, gt):
    
    img = Image.open(img_name)
    img = img.resize((gt.shape[3], gt.shape[2]), Image.ANTIALIAS)
    img.save("123.jpg", format=None)
    
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    
    return np_img

def compute_errors(gt, pred):
    #TODO: add batch process
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, 
                log_10=log_10)
    
    
if __name__ == "__main__":
    gt_path = "data/nyuv2/depths/562.png"
    pred_path = "imgs/562_pred.jpg"
    gt = read_gt_img(gt_path)
    gt = np.expand_dims(gt, axis=0)
    pred = read_pred_img(pred_path, gt)
    pred = np.mean(pred, axis=3)
    pred = np.expand_dims(pred, axis=0)

    results = compute_errors(gt, pred)
    pdb.set_trace()
    