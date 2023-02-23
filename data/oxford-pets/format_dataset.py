import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import shutil
import os
import json

tasks = ['seg', 'cls', 'det']

def get_seg_img(root, img_id):
    img_path = os.path.join(root, 'annotations/trimaps', '%s.png' % img_id)
    seg = Image.open(img_path).convert("RGB")
    seg = np.array(seg)
    seg -= 2
    seg_img = Image.fromarray(seg).convert("RGB")
    return seg_img

def get_bbox_img(root, img_id):
    xml_file = os.path.join(root, 'annotations/xmls', '%s.xml' % img_id)
    if os.path.exists(xml_file) == False:
        return None

    tree = ET.parse(xml_file)
    obj = tree.find('object')
    bndbox = obj.find('bndbox')

    bbox = [int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)]


    img_path = os.path.join(root, 'images', '%s.jpg' % img_id)
    img = Image.open(img_path).convert("RGB")
    box_img = img.copy()
    a = ImageDraw.ImageDraw(box_img)
    a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='red', width=8)

    del a

    return box_img

def get_class_img(img_id, target_name):
    img_path = os.path.join(root, 'images', '%s.jpg' % img_id)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype(font='YaMingTiC-2.ttf', size=20)
    font = ImageFont.truetype("keyboard.ttf", 26)
    draw.text(xy=(30,30),text=target_name, fill='red', font=font)

    del draw

    return img


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
    img = Image.open(img_path).convert("RGB")

    for task_type in tasks:


        target_name = ' '.join(img_id.split('_')[:-1]).strip()
        if task_type == 'seg':
            output_img = get_seg_img(root, img_id)
            prompt = {}
            prompt['edit'] = 'segment the {}'.format(target_name)

        elif task_type == 'cls':
            output_img = get_class_img(img_id, target_name)
            prompt = {}
            prompt['edit'] = 'what is the animal?'.format(target_name)

        else:
            output_img = get_bbox_img(root, img_id)
            if output_img is None:
                continue
            prompt = {}
            prompt['edit'] = 'detect the {}'.format(target_name)

        output_path = os.path.join(root, 'image_pairs', img_id + '_{}'.format(task_type))
        if os.path.exists(output_path) == False:
            os.mkdir(output_path)

        promt_file = open(os.path.join(output_path, 'prompt.json'), 'w')   
        promt_file.write(json.dumps(prompt))
        promt_file.close()

        img.save(output_path+'/{}_{}_0.jpg'.format(img_id, task_type))
        output_img.save(output_path + '/{}_{}_1.jpg'.format(img_id, task_type))

        seeds.append([img_id+'_{}'.format(task_type), [img_id+'_{}'.format(task_type)]])

    n +=1 
    if n % 100 == 0:
        print('{} images processed!'.format(n))

seed_file.write(json.dumps(seeds))

seed_file.close()
