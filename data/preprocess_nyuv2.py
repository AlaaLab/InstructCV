import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os 
from PIL import Image


def generate_img(f):
    f=h5py.File("nyu_depth_v2_labeled.mat")
    images=f["images"]
    images=np.array(images)
    
    path_converted='./nyu_images'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)
    
    from PIL import Image
    images_number=[]
    for i in range(len(images)):
        images_number.append(images[i])
        a=np.array(images_number[i])
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        iconpath='./nyu_images/'+str(i)+'.jpg'
        img.save(iconpath,optimize=True)

def generate_depth_map(f): 
    depths=f["depths"]
    depths=np.array(depths)
    
    path_converted='./nyu_depths/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)
    
    max = depths.max()
    print(depths.shape)
    print(depths.max())
    print(depths.min())
    
    depths = depths / max * 255
    depths = depths.transpose((0,2,1))
    
    print(depths.max())
    print(depths.min())
    
    for i in range(len(depths)):
        print(str(i) + '.png')
        depths_img= Image.fromarray(np.uint8(depths[i]))
        depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
        iconpath=path_converted + str(i)+'.png'
        depths_img.save(iconpath, 'PNG', optimize=True)


f=h5py.File("nyu_depth_v2_labeled.mat")
generate_depth_map(f)