import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets.folder import default_loader
import shutil
import os
import json


def get_seg_img(seg):
    seg = np.array(seg)
    seg -= 2
    seg_img = Image.fromarray(seg).convert("RGB")
    return seg_img


def get_cls_name():
    xml_file = os.path.join(dir, 'annotations/xmls', '%s.xml' % image_ids[i])
    tree = ET.parse(xml_file)
    obj = tree.find('object')
    bndbox = obj.find('bndbox')



n = 0
root = ''
seed_file = open(os.path.join(root, 'image_pairs', 'seeds.json'), 'w')
seeds = []
for line in open(os.path.join(root, 'annotations/trainval.txt')):
    line = line.strip()
    words = line.split(' ')
    img_id = words[0]
    cls_label = words[1]


    img_path = os.path.join(root, 'images', '%s.jpg' % img_id)
    seg_path = os.path.join(root, 'annotations/trimaps', '%s.png' % img_id)

    seg = Image.open(seg_path)

    seg_img = get_seg_img(seg)

    output_path = os.path.join(root, 'image_pairs', img_id)
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    promt_file = open(os.path.join(output_path, 'prompt.json'), 'w')

    seg_img.save(output_path + '/{}_seg_1.jpg'.format(img_id))
    
    seeds.append([img_id, [img_id+'_seg']])

    shutil.copyfile(img_path, output_path+'/{}_seg_0.jpg'.format(img_id))

    n +=1 
    if n % 100 == 0:
        print('{} images processed!'.format(n))

    prompt = {}
    prompt['edit'] = 'segment the {}'.format(' '.join(img_id.split('_')[:-1]).strip())
    #print(prompt)
    promt_file.write(json.dumps(prompt))

seed_file.write(json.dumps(seeds))

seed_file.close()
