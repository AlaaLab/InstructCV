# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Generate training dataset
# --------------------------------------------------------

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import shutil
import os
from random import shuffle
import json
import pdb
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
from fnmatch import fnmatch
import random
import copy
from numpy import asarray

#for ade20k
CLASSES = (
        'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

#for coco
CLASSES_COCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',#6
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',#11
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',#17
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',#24
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',#30
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',#35
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',#39
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',#46
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',#52
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',#58
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',#64
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',#69
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',#75
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')#80

Pet_CLASSES = ('Abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle','Bengal',#6
               'Birman', 'Bombay', 'boxer', 'British Shorthair', 'chihuahua', 'Egyptian Mau', 'english cocker spaniel',#13
               'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin',#18
               'keeshond', 'leonberger', 'Maine Coon', 'miniature pinscher', 'newfoundland', 'Persian',#24
               'pomeranian', 'pug', 'Ragdoll', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier',#31
               'shiba inu', 'Siamese', 'Sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier')#37

# COLOR = ((0, 0, 0), (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
#             (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
#             (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
#             (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
#             (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
#             (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
#             (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
#             (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
#             (255, 184, 6), (10, 255, 71), (255, 41, 10),(255, 0, 0))

COLOR   = ('red', 'green', 'blue', 'purple', 'white', 'black', 'AliceBlue', 'Aqua', 'Peru',#9
            'Brown', 'DarkGray', 'Gold', 'Violet', 'SlateBlue', 'Orange',#15
            'Maroon', 'LightSlateGray', 'Indigo','DarkKhaki', 'Coral','RosyBrown',#21
            'LightSalmon', 'Azure','Beige','CadetBlue','DarkBlue','Firebrick',#27
            'Silver','YellowGreen','LightPink','Snow','Sienna','Salmon','PowderBlue',#34
            'PeachPuff','DarkRed','Olive')#37

#for VOC
CLASSES_VOC = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "dining table", "dog",
               "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
               "train", "tvmonitor", "border")

COLOR_VOC = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
             [0, 128, 128], [128, 128, 128], [64, 0, 0], [
                 192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
             [192, 0, 128], [64, 128, 128], [192, 128, 128], [
                 0, 64, 0], [128, 64, 0], [0, 192, 0],
             [128, 192, 0], [0, 64, 128], [224, 224, 192]]


def get_bbox_prompt(cname):
    
    prompts = {}
    flag = random.randint(1,len(det_prompts)-1)
    prompt = det_prompts[flag]
    if prompt == '':
        assert "delete the empty line in det prompts txt file."
    prompt = copy.deepcopy(prompt)
    prompt = prompt.replace("%", cname)
    prompts['edit'] = prompt

    return prompts


def get_seg_prompt(cname):

    prompts = {}
    if len(seg_prompts) == 1:
        flag = 0
    else:
        flag = random.randint(1,len(seg_prompts)-1)
        
    prompt = seg_prompts[flag]
    if prompt == '':
        assert "delete the empty line in seg prompts txt file."
    prompt = copy.deepcopy(prompt)
    prompt = prompt.replace("%", cname)
    prompts['edit'] = prompt
    return prompts


def get_seg_ins_prompt():

    prompts = {}
    if len(ins_prompts) == 1:
        flag = 0
    else:
        flag = random.randint(1,len(ins_prompts)-1)

    prompt = ins_prompts[flag]
    prompts['edit'] = prompt
    return prompts


def get_depes_prompt():

    prompts = {}
    if len(dep_est_prompts) == 1:
        flag = 0
    else:
        flag = random.randint(1,len(dep_est_prompts)-1)

    prompt = dep_est_prompts[flag]
    
    if prompt == '':
        assert "delete the empty line in depes prompts txt file."
        
    prompts['edit'] = prompt
    return prompts


def get_cls_prompt(cls_name, pos_color, neg_color):
    
    prompts = {}
    if len(cls_prompts) == 1:
        flag = 0
    else:
        flag = random.randint(1,len(cls_prompts)-1)

    prompt = cls_prompts[flag]
    
    if prompt == '' or ' ':
        assert "delete the empty line in cls prompts txt file."
        
    prompt = copy.deepcopy(prompt)
    prompt = prompt.replace("*", pos_color)
    prompt = prompt.replace("%", cls_name)
    prompt = prompt.replace("#", neg_color)
    prompts['edit'] = prompt
    
    return prompts


def get_seg_img(root, img_id):
    
    img_path = os.path.join(root, 'annotations/trimaps', '%s.png' % img_id)
    seg = Image.open(img_path).convert("RGB")
    seg = np.array(seg)
    seg -= 2
    seg_img = Image.fromarray(seg).convert("RGB")
    
    return seg_img


def get_bbox_img(root, img_id, bbox, dataset):
    
    if dataset == 'oxford-pets':
        xml_file = os.path.join('./data/oxford-pets/annotations/xmls', '%s.xml' % img_id.replace('-', '_'))
        if os.path.exists(xml_file) == False:
            return None, bbox

        tree = ET.parse(xml_file)
        obj = tree.find('object')
        bndbox = obj.find('bndbox')

        bbox = [int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)]

        img_path = os.path.join('./data/oxford-pets/images', '%s.jpg' % img_id.replace('-', '_'))

        img = Image.open(img_path).convert("RGB")
        box_img = img.copy()
        # box_img = Image.new('RGB', img.size, (0,0,0))
        a = ImageDraw.ImageDraw(box_img)
        a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='blue', width=6)
        # a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill='white', outline='white', width=1)
    
    elif dataset == 'MSCOCO':
        img_path = os.path.join(root, 'train2017', '%s.jpg' % img_id)

        img = Image.open(img_path).convert("RGB")
        # box_img = Image.new('RGB', img.size, (0,0,0))

        # a = ImageDraw.ImageDraw(box_img)

        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        
        for box in bbox:

            a.rectangle(((box[0], box[1]), (box[2], box[3])), fill=None, outline="blue", width=4)

        del a
    
    
    elif dataset == 'VOC':
        img_path = os.path.join(root, 'JPEGImages', '%s.jpg' % img_id)
        img = Image.open(img_path).convert("RGB")
        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        for box in bbox:

            a.rectangle(((box[0], box[1]), (box[2], box[3])), fill=None, outline="red", width=5)
        del a

    return box_img, bbox

    
def get_class_img(img, color):
    
    cls_img = Image.new('RGB', img.size, color)
    
    return cls_img


def generate_sample(img, img_id, out_img, prompt, task_type):
    '''
    Args img: input/original images
         out_img: add bbox for det; mask for seg; depth img for DE .. 
         prompt: language instructions
         
    Return seed
    '''

    output_path = os.path.join(args.save_root, img_id + '_{}'.format(task_type))

    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    img.save(output_path+'/{}_{}_0.jpg'.format(img_id, task_type))
    # pdb.set_trace()
    out_img.save(output_path + '/{}_{}_1.jpg'.format(img_id, task_type))
    
    seed = [img_id+'_{}'.format(task_type), [img_id+'_{}'.format(task_type)]]

    promt_file = open(os.path.join(output_path, 'prompt.json'), 'w')
    promt_file.write(json.dumps(prompt))
    promt_file.close()

    return seed


def proc_oxford_pets_binary(oxford_pets_root, tasks):
    n = 0
    seeds = []

    for line in open(os.path.join(oxford_pets_root, 'annotations/trainval.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[2]
        
        if cls_label == '1':
            cls_name = 'cat'
        if cls_label == '2':
            cls_name = 'dog'
            
        clses = ['cat', 'dog']

        img_path = os.path.join(oxford_pets_root, 'images', '%s.jpg' % img_id)
        img = Image.open(img_path).convert("RGB")

        if cls_name == 'cat':
        #generate outputs
            for i in range(2):
                output_img = get_class_img(img, cls_name, "blue", is_pos=True)
                prompt = get_cls_prompt(cls_name, "blue")
                img_id = img_id + "_" + "blue"
                seed = generate_sample(img, img_id, output_img, prompt, 'cls_pos_{}'.format(i))
        
        if cls_name == 'dog':
            output_img = get_class_img(img, cls_name, "green", is_pos=True)
            prompt = get_cls_prompt(cls_name, "green")
            img_id = img_id + "_" + "green"
            seed = generate_sample(img, img_id, output_img, prompt, 'cls_pos')      
        
        n +=1 
        if n % 100 == 0:
            print('{} images processed!'.format(n))
        
    return seeds


def proc_oxford_pets_binary_ifelse(oxford_pets_root, num_loop, tasks):
    n = 0
    seeds = []

    for line in open(os.path.join(oxford_pets_root, 'annotations/trainval.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[2]
        
        if cls_label == '1':
            cls_name = 'cat'
        if cls_label == '2':
            cls_name = 'dog'
            
        clses = ['cat', 'dog']

        img_path = os.path.join(oxford_pets_root, 'images', '%s.jpg' % img_id)
        img = Image.open(img_path).convert("RGB")
        
        for cls in clses:
            if cls == cls_name: #positive sample
                
                #randomly generate two colors
                pos_c = random.choice(lcolor) #pos color
                neg_c = random.choice(lcolor) #neg color
                while pos_c == neg_c:
                    pos_c = random.choice(lcolor)
                    neg_c = random.choice(lcolor)

                #generate outputs
                output_img = get_class_img(img, pos_c)
                prompt = get_cls_prompt(cls_name, pos_c, neg_c)
                img_id = img_id + "_" + pos_c
                seed = generate_sample(img, img_id, output_img, prompt, 'cls_pos_{}'.format(num_loop))
        
            else: #negtive sample
                
                pos_c = random.choice(lcolor) #pos color
                neg_c = random.choice(lcolor) #neg color
                while pos_c == neg_c:
                    pos_c = random.choice(lcolor)
                    neg_c = random.choice(lcolor)
                
                output_img = get_class_img(img, neg_c)
                prompt = get_cls_prompt(cls, pos_c, neg_c)
                img_id = img_id + "_" + pos_c
                seed = generate_sample(img, img_id, output_img, prompt, 'cls_neg_{}'.format(num_loop))
                
        
        n +=1 
        if n % 100 == 0:
            print('{} images processed!'.format(n))
        
    return seeds
    
      
def proc_oxford_pets_finegrained(oxford_pets_root, tasks):

    n = 0
    seeds = []
    
    for line in open(os.path.join(oxford_pets_root, 'annotations/trainval.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[1]

        target_name = ' '.join(img_id.split('_')[:-1]).strip()
        clses[cls_label] = target_name

    for line in open(os.path.join(oxford_pets_root, 'annotations/trainval.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[1]

        target_name = ' '.join(img_id.split('_')[:-1]).strip()

        img_path = os.path.join(oxford_pets_root, 'images', '%s.jpg' % img_id)
        img = Image.open(img_path).convert("RGB")

        for task_type in tasks:
            if task_type == 'seg':
                output_img = get_seg_img(oxford_pets_root, img_id)
                if output_img is None:
                    assert "seg output image cannot be nonetype"
                prompt = {}
                prompt['edit'] = 'segment the {}'.format(target_name)
                seed = generate_sample(img, img_id, output_img, prompt, 'seg')
                seeds.append(seed)

            elif task_type == 'cls':
                
                for cls in clses:
                    # pdb.set_trace()
                    if cls == cls_label:
                        ## randomly set color
                        # c = random.choice(lcolor)
                        
                        for i in range(len(lcolor)):

                            # c = lcolor[i]
                            # color = colors[c]
                            c = 'white'
                        
                            ## specific set color
                            # color = pet_to_color[target_name] # {cat:red}

                            output_img = get_class_img(img, target_name, c, is_pos=True)
                            if output_img is None:
                                assert "cls output image cannot be nonetype"
                                
                            prompt = {'edit': 'show {} if contains {} else black'.format(c, target_name)}
                            # prompt = {'edit':'classify this image'}
                            # fixed prompt:
                            # prompt = {'edit': 'show the corresponding color of this {}'.format(target_name)}
                            img_id2 = img_id + "_" + c
                            seed = generate_sample(img, img_id2, output_img, prompt, task_type + '_pos')
                            seeds.append(seed)
                    
                    # if cls == cls_label:
                    #     continue
                    else:
                        if random.random() > neg_sample_rate:  # 负采样率
                            continue
                        nname = clses[cls]
                        
                        # c = random.choice(lcolor)
                        # color = colors[c]
                        c='black'
                        
                        # color = pet_to_color[nname]
                        
                        output_img = get_class_img(img, nname, c, is_pos=False)
                        prompt = {'edit': 'show white if contains {} else black'.format(nname)}
                        # prompt = {'edit':'classify this image'}
                        seed = generate_sample(img, img_id, output_img, prompt, task_type+'_neg_{}'.format(nname))
                        seeds.append(seed)
                    n += 1
                    if n % 100 == 0:
                        print('{} images processed!'.format(n))
                    continue
            
            else:
                    
                output_img, bbox = get_bbox_img(img_id, 1, dataset='oxford-pets')
                    
                if output_img is None:
                    continue
                
                prompt  = get_bbox_prompt(target_name)
                
                seed = generate_sample(img, img_id, output_img, prompt, task_type)
                seeds.append(seed)

                output_path = os.path.join(args.save_root, img_id + '_det')
                bbox_info = {'bbox': bbox}
                bbox_file = open(os.path.join(output_path, 'bbox.json'), 'w')
                bbox_file.write(json.dumps(bbox_info))
                bbox_file.close()

        n +=1 
        if n % 100 == 0:
            print('{} images processed!'.format(n))
        
    return seeds


def preproc_coco(root):
    
    print('begin to pre-process coco dataset...')
    clses                   = {}
    coco_path               = os.path.join(root, 'annotations/instances_train2017.json')
    coco_fp                 = open(coco_path)
    anno_js                 = json.loads(coco_fp.readline())

    for cate in anno_js['categories']:
        
        cid                 = cate['id']
        cname               = cate['name']
        clses[cid]          = cname


    for key in anno_js:
        print(key)

    img_info = {}
    coco_anno = anno_js['annotations']

    for anno in coco_anno:
        image_id = str(anno['image_id'])
        box = list(anno['bbox'])
        cbox = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        cid = anno['category_id']
        segmentation = anno['segmentation']
        iscrowd = anno['iscrowd']

        if image_id not in img_info:
            img_info[image_id] = {}

        if cid not in img_info[image_id]:
            img_info[image_id][cid] = {'bbox': [], 'segmentation': []}

        img_info[image_id][cid]['bbox'].append(cbox)
        if iscrowd == 0:
            img_info[image_id][cid]['segmentation'].append(segmentation)

    return img_info, clses


def preproc_voc(root):
    '''
    Return: 
        -- img_info: dict. {'id':{'bbox':... , 'segmentation':...}}
        -- clses: dict. all classes
        -- img_id_map: dict. map relationship between id (1-1499) and img name 
    '''
    
    print('begin to pre-process voc dataset...')
    clses, img_id_map       = {},{}
    voc_path                = os.path.join(root, 'data/train.json')
    voc_fp                  = open(voc_path)
    anno_js                 = json.loads(voc_fp.readline())

    for cate in anno_js['categories']:
        cid                 = cate['id']
        cname               = cate['name']
        clses[cid]          = cname


    for key in anno_js:
        print(key)
    
    for item in anno_js['images']:
        iid                 = item['id']
        fname               = item['file_name']
        img_id_map[fname]   = iid
        
    img_info = {}
    coco_anno = anno_js['annotations']

    for anno in coco_anno:
        image_id = str(anno['image_id'])
        box = list(anno['bbox'])
        cbox = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        cid = anno['category_id']
        segmentation = anno['segmentation']
        iscrowd = anno['iscrowd']

        if image_id not in img_info:
            img_info[image_id] = {}

        if cid not in img_info[image_id]:
            img_info[image_id][cid] = {'bbox': [], 'segmentation': []}

        img_info[image_id][cid]['bbox'].append(cbox)
        if iscrowd == 0:
            img_info[image_id][cid]['segmentation'].append(segmentation)

    return img_info, clses, img_id_map


def proc_coco(coco_root, tasks):
    
    print('begin to process coco dataset...')
    
    seeds           = []
    n               = 0
    img_info, clses = preproc_coco(coco_root)
    
    #----------------split process----------------
    image_list=[]
    for image_id in img_info:
        image_list.append(image_id)
    image_list = sorted(image_list)
    print("len(image_list)", len(image_list))
    image_list = image_list[0:20000]
    # image_list = image_list[20000:40000]
    # image_list = image_list[40000:60000]
    # image_list = image_list[60000:80000]
    # image_list = image_list[80000:100000]
    # image_list = image_list[100000:len(image_list)]
    for image_id in image_list:
    #----------------split process----------------
    
    ## origin
    # for image_id in img_info:#image_id:355677
        
        for cid in img_info[image_id]:
            
            cname               = clses[cid] #target_name
            cname_id            = list(img_info[str(image_id)].keys())
            ncls_perimg         = [] # store g.t class names
            
            for i in cname_id:
                cname_           = clses[i]
                ncls_perimg.append(cname_)
            
            img_id              = image_id.zfill(12) #000000355677
            img_path            = os.path.join(coco_root, 'train2017/{}.jpg'.format(img_id))
            img                 = Image.open(img_path).convert("RGB")
            
            for task in tasks:
                
                if task == "det":
                    
                    bbox  = img_info[image_id][cid]['bbox']
                    
                    count = 0
                        
                    det_img, bbox = get_bbox_img(coco_root, img_id, bbox, dataset='MSCOCO')
                
                    prompt  = get_bbox_prompt(cname)
                    # prompt = {'edit': 'detect the {}'.format(cname)}
                    
                    # check if 3 channels
                    try:
                        r, g, b = det_img.split()
                    except Exception:
                        print("not 3 channels:", img_id)
                    
                    seed = generate_sample(img, img_id, det_img, prompt, task_type='det_{}'.format(cname))
                    seeds.append(seed)
                    
                    output_path = os.path.join(args.save_root, img_id + '_det_{}'.format(cname))
                    bbox_info = {'bbox': bbox}
                    bbox_file = open(os.path.join(output_path, 'bbox.json'), 'w')
                    bbox_file.write(json.dumps(bbox_info))
                    bbox_file.close()
                    count    += 1
                        
                    
                elif task == 'cls':
                    ## origin
                    c1 = random.choice(lcolor)
                    c2 = random.choice(lcolor)
                    while c1 == c2:
                        c1 = random.choice(lcolor)
                        c2 = random.choice(lcolor)
                    color = colors[c1]
                    
                    ##modified postive
                    # c     = 'white'
                    # color = (255,255,255)
                    
                    for cls_name in ncls_perimg:
                        
                        cls_img = get_class_img(img, cls_name, color, is_pos=True)
                
                        prompt = {'edit': 'show {} if the picture contain {}, otherwise show {}'.format(c1, cls_name, c2)}
                
                        seed = generate_sample(img, img_id, cls_img, prompt, task_type='cls_{}_pos'.format(cls_name))
                        seeds.append(seed)
                    
                    ## negative
                    for nname in CLASSES_COCO:
                        if nname in ncls_perimg:
                            # print("it is a posible sample:{}".format(nname))
                            continue
                        
                        if random.random() > neg_sample_rate:
                            continue
                        
                        # print("it is a negative sample:{}".format(nname))
                        
                        ##origin
                        c1 = random.choice(lcolor)
                        c2 = random.choice(lcolor)
                        while c1 == c2:
                            c1 = random.choice(lcolor)
                            c2 = random.choice(lcolor)
                        color = colors[c2]
                        
                        ##modified
                        # color = (0,0,0)
                        
                        output_img = get_class_img(img, nname, color, is_pos=False)

                        prompt = {'edit': 'show {} if the picture contain {}, otherwise show {}'.format(c1, nname, c2)}
                        
                        seed = generate_sample(img, img_id, output_img, prompt, task_type='cls_{}_neg_{}'.format(cname, nname))

                else:
                    h, w = img.size
                    gt = np.zeros((w, h), dtype=np.uint8)
                    for seg in img_info[image_id][cid]['segmentation']:
                        for s in seg:
                            s = np.array(s).reshape(-1, 2)     # [n_points, 2]
                            cv2.fillPoly(gt, s.astype(np.int32)[np.newaxis, :, :], (255, 255, 255))

                    prompt = get_seg_prompt(cname)
                    
                    seg_img = Image.fromarray(cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)).convert("RGB")
                    seed = generate_sample(img, img_id, seg_img, prompt, task_type='seg_{}'.format(cname))

                n +=1 
                if n % 100 == 0:
                    print('{} images processed!'.format(n))
        
    return


def proc_nyuv2_all(nyuv2_root):

    print('begin to process NYU_V2 training dataset...')
    
    seeds = []
    prompt = {}
    n = 0
    train_txt_path = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/nyu_mdet/nyu_train.txt'
    
    with open(train_txt_path) as file:  

        for line in file:
            
            img_path_part   = line.strip().split(" ")[0] # /kitchen_0028b/rgb_00045.jpg
            img_path_part   = img_path_part[1:len(img_path_part)] # kitchen_0028b/rgb_00045.jpg
            
            file_name       = img_path_part.split("/")[0] # kitchen_0028b
            
            if fnmatch(file_name.split("_")[1], "0*"): 
                cls             = file_name.split("_")[0]
            else:
                cls             = file_name.split("_")[0] + " " +file_name.split("_")[1]
            
            img_name        = img_path_part.split("/")[1] # rgb_00045.jpg
            img_path        = os.path.join(nyuv2_root, img_path_part)
            img_id          = file_name + "_" + img_name.split(".")[0] # kitchen_0028b_rgb_00045
            dep_path_part = file_name + "/vli_depth_" + img_name.split("_")[-1].replace("jpg","png") # kitchen_0028b/vli_depth_00045.jpg
            dep_path      = os.path.join(nyuv2_root, dep_path_part)
            
            img         = Image.open(img_path).convert("RGB")
            depth_img   = Image.open(dep_path).convert("RGB")
            
            prompt      = get_depes_prompt()

            seed = generate_sample(img, img_id, depth_img, prompt, task_type="depes")
            seeds.append(seed)
            
            n += 1 
            if n % 1000 == 0:
                print('{} images processed!'.format(n))
    
    return seeds


def proc_nyuv2(nyuv2_root):

    print('begin to process NYU_V2 training dataset...')
    
    seeds               = []
    prompt              = {}
    n = 0
    
    img_path            = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/nyuv2_labeled/nyu_images'
    depth_path          = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/nyuv2_labeled/nyu_depths'
    imgs                = os.listdir(img_path)
    
    for img_n in imgs: #0.jpg
        
        line  = img_n.strip()
        word  = line.split('.')
        img_id = word[0] #0
    
        img_path_        = os.path.join(img_path, '%s.jpg' % img_id)
        depth_path_      = os.path.join(depth_path, '%s.png' % img_id)
        
        img             = Image.open(img_path_).convert("RGB")
        depth_img       = Image.open(depth_path_).convert("RGB")
        
        prompt['edit'] = 'Estimate the depth of this image'
        # prompt = get_depth_prompt()
        
        seed = generate_sample(img, img_id, depth_img, prompt, task_type="depes")
        seeds.append(seed)
    
    return seeds


def proc_ade20k(ade_root):
    
    print('begin to process ade20k training dataset...')
    
    seeds = []
    prompt = {}
    n = 0
    
    file_ade = os.listdir(ade_root) #file_ade: [cultural,...]
    for file in file_ade: #file: cultural
        file_in_ade = os.listdir(os.path.join(ade_root, file)) #file_in_ade= [apse,...]
        for file_in in file_in_ade: #file_in: apse_indoor
            
            file_name_li = os.listdir(os.path.join(ade_root, file, file_in))
            
            img_path_dic = {}
            seg_path_dic = {}
            img_li = []

            for file_name in file_name_li:
                
                img_id = file_name.split(".")[0]

                if fnmatch(file_name, "*.jpg"):
                    img_path = os.path.join(ade_root, file, file_in, file_name)
                    img_path_dic[img_id] = img_path
                    img_li.append(img_id)
                    
                    img = Image.open(img_path).convert("RGB")
                 
                if fnmatch(file_name, "*_seg.png"):
                    
                    seg_path = os.path.join(ade_root, file, file_in, file_name)
                    seg_path_dic[img_id] = seg_path
                    seg_img = Image.open(seg_path).convert("RGB")
                
            
            for id in img_li:
                img_path = img_path_dic[id]
                seg_path = seg_path_dic[id+"_seg"]
                img = Image.open(img_path).convert("RGB")
                seg_img = Image.open(seg_path).convert("RGB")
                prompt  = get_seg_prompt(cname="image")
                seed = generate_sample(img, id, seg_img, prompt, task_type="seg")
                seeds.append(seed)
                
                n +=1 
                if n % 100 == 0:
                    print('About {} images processed!'.format(n*5))

    return seeds


def proc_imagenet(imagenet_root):
    
    # IMAGENET_MAP = {"n04613696": "mongolian yurt yellow", "n01440764": "fish blue", 
    #                 "n04476259": "Plate green", "n04606251": "shipwrecks red", 
    #                 "n06359193": "website purple"}
    
    IMAGENET_MAP = {"n04613696": "yurt yellow green", "n01440764": "fish blue red"}
    
    cls_num      = len(IMAGENET_MAP.keys())
    clses        = list(IMAGENET_MAP.keys())
    
    for item in IMAGENET_MAP.keys():
        
        split           = IMAGENET_MAP[item].split(" ")
        file_path       = os.path.join(imagenet_root, item)
        image_list      = os.listdir(file_path)
        image_list      = sorted(image_list)
        color_neg       = split[-1]
        color_pos       = split[-2]
        
        for i in range(cls_num):
            
            if clses[i] == item: # pos

                for m, img_pa in enumerate(image_list): #len(image_list) = 9, i 0,1,2,...8
                    
                    cls_name    = IMAGENET_MAP[item].split(" ")[0]
                    img_path    = os.path.join(file_path, img_pa)
                    img         = Image.open(img_path).convert('RGB')
                    img_id      = img_pa.split(".")[-2] #00000293
                
                    output_img = get_class_img(img, color_pos)
                    if output_img is None:
                        assert "cls output image cannot be nonetype"

                    prompt = get_cls_prompt(cls_name, color_pos, color_neg)
                    
                    img_id2 = img_id + "_" + color_pos
                    seed = generate_sample(img, img_id2, output_img, prompt, cls_name + '_cls_pos')
                        
            else: # neg
                
                for m, img_pa in enumerate(image_list): #len(image_list) = 9, i 0,1,2,...8
                    
                    cls_name    = IMAGENET_MAP[clses[i]].split(" ")[0]
                    img_path    = os.path.join(file_path, img_pa)
                    img         = Image.open(img_path).convert('RGB')
                    img_id      = img_pa.split(".")[-2] #00000293
                
                    output_img  = get_class_img(img, color_neg)
                    
                    if output_img is None:
                        assert "cls output image cannot be nonetype"

                    prompt      = get_cls_prompt(cls_name, color_pos, color_neg)
                    
                    img_id2     = img_id + "_" + color_neg
                    seed        = generate_sample(img, img_id2, output_img, prompt, cls_name + '_cls_neg')
                
    return
                
            
def proc_adechan2016_sem(ade_root, cls_ade_dict):
    
    print('begin to process ade20k training dataset...')
    
    seeds = []
    prompt = {}
    n = 0
    
    img_list = os.listdir(os.path.join(ade_root, "images/training"))
    
    #part
    img_list = sorted(img_list)
    print("len(image_list)", len(img_list))
    img_list = img_list[0:5000]
    # img_list = img_list[15000:len(img_list)]
    #######
    
    for img_name in img_list:
        
        img_path = os.path.join(ade_root, "images/training", img_name)
        seg_path = os.path.join(ade_root, "annotations/training", img_name.split(".")[0]+".png")
        anno = Image.open(seg_path)
        anno = np.array(anno)
        
        clses = np.unique(anno)
        
        for cls in clses: # e.g., cls=1
            img = Image.open(img_path).convert('RGB') #original image
            # seg_img = Image.new('RGB', img.size, (0,0,0))
            
            seg_img = Image.new('RGB',(img.size[0],img.size[1]), color=0)
            seg_img = np.array(seg_img)

            #find where equals cls in anno
            r, c = np.where(anno == cls) #r,c are arraries
            for i in range(len(r)):
                seg_img[r[i],c[i],:] = (255,255,255)

            seg_img = Image.fromarray(seg_img).convert('RGB')
            # pdb.set_trace()
            
            cls_name = cls_ade_dict[cls]
            cls_name_wospace = cls_name
            if " " in cls_name:
                cls_name_wospace = cls_name.replace(" ", "_")
            
            if cls_name == "background":
                continue

            prompt  = get_seg_prompt(cname=cls_name)
            # prompt = {'edit': 'segment the {}'.format(cls_name)}
            
            # check if 3 channels
            try:
                r, g, b = seg_img.split()
                r1, g1, b1 = img.split()
            except Exception:
                print("not 3 channels:", img_name)
                
            seed = generate_sample(img, img_name.split(".")[0]+"_"+cls_name_wospace, seg_img, prompt, task_type="seg")
            seeds.append(seed)
    
            n +=1 
            if n % 100 == 0:
                print('About {} images processed!'.format(n))

    return seeds


def proc_adechan2016_ins(ade_root):
    
    print('begin to process ade20k instance seg training dataset...')
    
    seeds   = []
    prompt  = {}
    n       = 0
    colors  = [(0,0,0)]

    for i in range(255):
        colors.append((random.randrange(1, 255, 1), random.randrange(1, 255, 1), random.randrange(1, 255, 1)))

    colors = np.array(colors)
    
    img_list = os.listdir(os.path.join(ade_root, "images/training"))
    
    #part
    img_list = sorted(img_list)
    print("len(image_list)", len(img_list))
    img_list = img_list[0:10000]
    # img_list = img_list[10000:len(img_list)]
    #######
    
    for img_name in img_list:
        
        img_path = os.path.join(ade_root, "images/training", img_name)
        seg_path = os.path.join(ade_root, "annotations_instance/training", img_name.split(".")[0]+".png")
        
        # seg_img = np.array(Image.open(img_path))
        seg_img = Image.open(img_path).convert('RGB')
        im_label = Image.open(seg_path).convert('L')
        w, h = im_label.size
        im_label = np.array(im_label)
        im_label = colors[im_label]
        im_label = Image.fromarray(np.uint8(im_label)).convert('RGB')
        # new_img = np.where(im_label == 0, seg_img[:,:], im_label[:,:])
        # new_img = Image.fromarray(new_img.astype(np.uint8))
        
        prompt  = get_seg_ins_prompt()
        
        seed = generate_sample(seg_img, img_name.split(".")[0], im_label, prompt, task_type="ins_seg")

        n +=1 
        if n % 100 == 0:
            print('About {} images processed!'.format(n))

    return seeds


def get_colors(image_path):
    
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    colors = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            colors.add((r[i][j], g[i][j], b[i][j]))
    return list(colors)


def color_replace(img, color):
    
    color = color[::-1]
    img = cv2.imread(img)
    lower = np.array(color)
    upper = np.array(color)
    mask = cv2.inRange(img, lower, upper)
    img[mask > 0] = [255, 255, 255]
    img[mask == 0] = [0, 0, 0]
    return img


def proc_vocdataset(voc_root):
    
    img_ins_root = os.path.join(voc_root, "SegmentationObject_new")
    
    for img_name in os.listdir(img_ins_root): #2007_000032.png
        
        img_name_jpg = img_name.replace("png","jpg")
        img_path = os.path.join(voc_root, "JPEGImages",img_name_jpg)
        img_ins_path = os.path.join(img_ins_root, img_name)
    
        img_id = img_name.split(".")[0]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(img_ins_path).convert('RGB')#also seg_path
        
        prompt  = get_seg_ins_prompt()
        
        seed = generate_sample(img, img_id, label, prompt, task_type="seg")
    return
    

def proc_fs(fs_root, split):
    
    for line in open(os.path.join(fs_root, split)):
        line        = line.strip()
        
        if line.split(".")[-1] == "png":
            continue
        
        cname       = line.split("/")[0].replace("_", " ")
        img_id      = cname + "_" + line.split("/")[-1].split(".")[0]
        
        img_p       = os.path.join(fs_root, line)
        label_p     = img_p.replace("jpg","png")
        
        img         = Image.open(img_p).convert('RGB')
        label       = Image.open(label_p).convert('RGB')
        
        prompt      = get_bbox_prompt(cname)
        seed = generate_sample(img, img_id, label, prompt, task_type="seg")
    
    return
        
    
def prompts_chat():
    '''
    Use ChatGPT to generate various prompts.
    '''
    # prompt_chat = []
    flag = {} #key: path to prompts.json; value: prompt
    root_img_pair = "image_pairs"
    file_all = os.listdir(root_img_pair)
    
    for file_n in file_all: # file_n: 0_depes
        
        if file_n == 'seeds.json':
            continue
        
        prompt_path = os.path.join(root_img_pair, file_n, 'prompt.json')
        
        with open(prompt_path,'r',encoding = 'utf-8') as fp:
            prompts = json.load(fp)
            prompt = prompts["edit"]
            flag[prompt_path] = prompt
        fp.close()
    
    num_prompts = len(flag)
    
    prompt_chat = list(flag.values())
    prompt_chat_loc = list(flag.keys())
    
    return prompt_chat, prompt_chat_loc


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="coco", type=str)
    parser.add_argument("--save_root", default="./image_pairs", type=str)
    parser.add_argument("--data_root", default="", type=str)
    parser.add_argument("--oxford_pets_root", default='./data/oxford-pets', type=str)
    parser.add_argument("--coco_root", default='./data/coco', type=str)
    parser.add_argument("--nyuv2_root", default='./data/nyu_mdet', type=str)
    parser.add_argument("--ade_root", default='./data/ADEChallengeData2016', type=str)
    parser.add_argument("--imagenet_root", default='./data/imagenet_train', type=str)
    parser.add_argument("--voc_root", default='./data/VOCdevkit/VOC2012', type=str)
    parser.add_argument("--fs_root", default='./data/fewshot_data/fewshot_data', type=str)
    args = parser.parse_args()
    
    cls_ade_dict, pet_to_color, clses                          = {}, {}, {}
    neg_sample_rate                                            = 1000 # for cls. 0 means no negtive sample
    num_seg, num_det, num_dep_est, num_cls, num_ins            = 0, 0, 0, 0, 0
    seg_prompts, det_prompts, dep_est_prompts, cls_prompts, ins_prompts   = {}, {}, {}, {}, {}
    
    for i in range(len(CLASSES)):
        
        cls_ade_dict[i]                         = CLASSES[i]
    
    # get colors for cls
    # colors                                      = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 
    #                                                 'purple':(128,0,128), 'white':(255,255,255), 'AliceBlue':(240,248,255), 'Aqua': (0,255,255), 'Peru':(205, 133, 63),
    #                                                 'Brown':(165,42,42), 'DarkGray':(169,169,169), 'Gold':(255,215,0),
    #                                                 'Violet':(238,130,238), 'SlateBlue':(230,230,250), 'Orange':(255,128,0),
    #                                                 'Maroon':(128,0,0), 'LightSlateGray':(119,136,153), 'Indigo':(75,0,130),
    #                                                 'DarkKhaki':(189,183,107), 'Coral':(255,127,80),'RosyBrown':(188,143,143),
    #                                                 'LightSalmon':(255,160,122), 'Azure':(240,255,255),'Beige':(245,245,220),
    #                                                 'CadetBlue':(95,158,160),'DarkBlue':(0,0,139),'Firebrick':(178,34,34),
    #                                                 'Silver':(192,192,192),'YellowGreen':(154,205,50),'LightPink':(255,182,193),
    #                                                 'Snow':(255,250,250),'Sienna':(160,82,45),'Salmon':(250,128,114),
    #                                                 'PowderBlue':(176,224,230),'PeachPuff':(255,218,155),'DarkRed':(139,0,0)}
    colors                                      = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 
                                                    'purple':(128,0,128), 'white':(255,255,255), 
                                                    'Brown':(165,42,42), 'Gold':(255,215,0),'Beige':(245,245,220),
                                                    'Violet':(238,130,238), 'Orange':(255,128,0),
                                                    'Maroon':(128,0,0), 'Indigo':(75,0,130), 'Aqua': (0,255,255),
                                                    'Coral':(255,127,80),'RosyBrown':(188,143,143),'Peru':(205, 133, 63)}
    # colors                                      = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255)}
    
    lcolor                                      = list(colors.keys())

    for i, pet_name in enumerate(Pet_CLASSES):
        c                                       = COLOR[i]
        pet_to_color[pet_name]                  = c #{cat: red,...}

    if os.path.exists(args.save_root)  == False:
        os.mkdir(args.save_root)
        
        
    
    #generate prompts dict
    
    # with open("./data/seg_prompts.txt") as file:
    #     for item_seg in file:
    #         sen_seg = item_seg.strip()
    #         if sen_seg == '':
    #             continue
    #         seg_prompts[num_seg] = sen_seg
    #         num_seg += 1
    #     seg_prompts_new = seg_prompts.copy()
        
    #     while len(seg_prompts_new.values()) <= 5000:
            
    #         for prompt in seg_prompts_new.values():
    #             seg_prompts_tmp = seg_prompts_new.copy()
    #             output = Rephrase(prompt).do()
    #             for i in range(0,3):
    #                 output_i = output[i].replace("percentage", "%")
    #                 seg_prompts_tmp[num_seg] = output_i
    #                 num_seg += 1
                    
    #         seg_prompts_new_ = {}
    #         dict_key_ls = list(seg_prompts_tmp.keys())
    #         random.shuffle(dict_key_ls)
    #         for key in dict_key_ls:
    #             seg_prompts_new_[key] = seg_prompts_tmp.get(key)
    #         seg_prompts_new = seg_prompts_new_.copy()
    

    # json_file = open('dataset_creation/seg_prompt.json', 'w')
    # json_file.write(json.dumps(seg_prompts_new))
    # json_file.close()
            
    
    # with open("./data/det_prompts.txt") as file:
    #     for item_det in file:
    #         sen_det = item_det.strip()
    #         det_prompts[num_det] = sen_det
    #         num_det += 1
            
    #     det_prompts_new = det_prompts.copy()
        
    #     while len(det_prompts_new.values()) <= 5000:
            
    #         for prompt in det_prompts_new.values():
    #             det_prompts_tmp = det_prompts_new.copy()
    #             output = Rephrase(prompt).do()
    #             for i in range(0,3):
    #                 output_i = output[i].replace("percentage", "%")
    #                 det_prompts_tmp[num_det] = output_i
    #                 num_det += 1
                    
    #         det_prompts_new_ = {}
    #         dict_key_ls = list(det_prompts_tmp.keys())
    #         random.shuffle(dict_key_ls)
    #         for key in dict_key_ls:
    #             det_prompts_new_[key] = det_prompts_tmp.get(key)
    #         det_prompts_new = det_prompts_new_.copy()
        
    # json_file = open('dataset_creation/det_prompt.json', 'w')
    # json_file.write(json.dumps(det_prompts_new))
    # json_file.close()

    with open("./data/seg_prompts_rp.txt") as file:
        for item_seg in file:
            sen_seg = item_seg.strip()
            seg_prompts[num_seg] = sen_seg
            num_seg += 1
    
    with open("./data/det_prompts_rp.txt") as file:
        for item_det in file:
            sen_det = item_det.strip()
            det_prompts[num_det] = sen_det
            num_det += 1
            
    with open("./data/dep_est_prompts_rp.txt") as file:
        for item_dep_est in file:
            sen_dep_est = item_dep_est.strip()
            dep_est_prompts[num_dep_est] = sen_dep_est
            num_dep_est += 1
            
    with open("./data/cls_prompts.txt") as file:
        for item_cls_est in file:
            sen_cls_est = item_cls_est.strip()
            cls_prompts[num_cls] = sen_cls_est
            num_cls += 1
    
    with open("./data/seg_prompts_instance.txt") as file:
        for item_seg_ins in file:
            sen_seg_ins = item_seg_ins.strip()
            ins_prompts[num_ins] = sen_seg_ins
            num_ins += 1
    
    tasks = ['det']
    
    if fnmatch(args.dataset, "coco"):
        proc_coco(args.coco_root, tasks)
    
    elif fnmatch(args.dataset, "oxford_pets_fine"):
        proc_oxford_pets_finegrained(args.oxford_pets_root, tasks)
        
    elif fnmatch(args.dataset, "oxford_pets_binary"):
        loop = 5
        for num_loop in range(loop):
            proc_oxford_pets_binary_ifelse(args.oxford_pets_root, num_loop, tasks)

    elif fnmatch(args.dataset, "nyuv2"):
        proc_nyuv2_all(args.nyuv2_root)
    
    elif fnmatch(args.dataset, "ade20k_sem"):
        proc_adechan2016_sem(args.ade_root, cls_ade_dict)
        
    elif fnmatch(args.dataset, "ade20k_ins"):
        proc_adechan2016_ins(args.ade_root)
        
    elif fnmatch(args.dataset, "imagenet"):
        proc_imagenet(args.imagenet_root)
    
    elif fnmatch(args.dataset, "voc"):
        proc_vocdataset(args.voc_root)
    
    elif fnmatch(args.dataset, "fs"):
        proc_fs(args.fs_root, "test_part2.txt")


