import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import shutil
import os
import json
#from pycocotools.coco import COCO
import cv2
import random

tasks = ['det']

neg_sample_rate=0.1 # 负采样率

clses = {}

def get_seg_img(root, img_id):
    img_path = os.path.join(root, 'oxford-pets/annotations/trimaps', '%s.png' % img_id)
    seg = Image.open(img_path).convert("RGB")
    seg = np.array(seg)
    seg -= 2
    seg_img = Image.fromarray(seg).convert("RGB")
    return seg_img

def get_bbox_img(root, img_id, bbox, dataset):
    if dataset == 'oxford-pets':
        xml_file = os.path.join(root, 'oxford-pets/annotations/xmls', '%s.xml' % img_id)
        if os.path.exists(xml_file) == False:
            return None

        tree = ET.parse(xml_file)
        obj = tree.find('object')
        bndbox = obj.find('bndbox')

        bbox = [int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)]

        img_path = os.path.join(root, 'oxford-pets/images', '%s.jpg' % img_id)

        img = Image.open(img_path).convert("RGB")
        #box_img = img.copy()
        box_img = Image.new('RGB', img.size, (0,0,0))
        a = ImageDraw.ImageDraw(box_img)
        #a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='red', width=6)
        a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill='red', outline='red', width=1)
    
    elif dataset == 'MSCOCO':
        img_path = os.path.join(root, 'coco/train2017', '%s.jpg' % img_id)

        img = Image.open(img_path).convert("RGB")
        # box_img = img.copy()
        box_img = Image.new('RGB', img.size, (0,0,0))
        a = ImageDraw.ImageDraw(box_img)
        for box in bbox:
            #a.rectangle(box, fill=None, outline='red', width=4)
            a.rectangle(box, fill='red', outline='red', width=1)

    del a

    return box_img

def get_class_img(img, target_name, cls_label, is_pos):

    if is_pos:
        cls_img = Image.new('RGB', img.size, (255,255,255))
    else:
        cls_img = Image.new('RGB', img.size, (0,0,0))
    return cls_img

def generate_sample(root, img, out_img, prompt, task_type):

    output_path = os.path.join(root, 'image_pairs', img_id + '_{}'.format(task_type))

    if os.path.exists(output_path) == False:
        os.mkdir(output_path)


    img.save(output_path+'/{}_{}_0.jpg'.format(img_id, task_type))
    out_img.save(output_path + '/{}_{}_1.jpg'.format(img_id, task_type))
    seed = [img_id+'_{}'.format(task_type), [img_id+'_{}'.format(task_type)]]

    promt_file = open(os.path.join(output_path, 'prompt.json'), 'w')
    promt_file.write(json.dumps(prompt))
    promt_file.close()

    return seed


n = 0
root = ''
sample_path =  os.path.join(root, 'image_pairs')
if os.path.exists(sample_path) == False:
    os.mkdir(sample_path)

seed_file = open(os.path.join(root, 'image_pairs', 'seeds.json'), 'w')
seeds = []
for line in open(os.path.join(root, 'oxford-pets/annotations/trainval.txt')):
    line = line.strip()
    words = line.split(' ')
    img_id = words[0]
    cls_label = words[1]

    target_name = ' '.join(img_id.split('_')[:-1]).strip()
    clses[cls_label] = target_name

pet_cls_num = len(clses)
coco_path = os.path.join(root, 'coco/annotations/instances_train2017.json')
coco_fp = open(coco_path)
anno_js = json.loads(coco_fp.readline())

for cate in anno_js['categories']:
    cid = cate['id'] + pet_cls_num
    cname = cate['name']
    clses[cid] = cname

print(clses)


# for line in open(os.path.join(root, 'oxford-pets/annotations/trainval.txt')):
#     line = line.strip()
#     words = line.split(' ')
#     img_id = words[0]
#     cls_label = words[1]

#     target_name = ' '.join(img_id.split('_')[:-1]).strip()

#     img_path = os.path.join(root, 'oxford-pets/images', '%s.jpg' % img_id)
#     img = Image.open(img_path).convert("RGB")

#     for task_type in tasks:

#         if task_type == 'seg':
#             output_img = get_seg_img(root, img_id)
#             prompt = {}
#             prompt['edit'] = 'segment the {}'.format(target_name)

#         elif 0 and task_type == 'cls':
#             output_img = get_class_img(img, target_name, cls_label, is_pos=True)
#             prompt = {'edit': 'if the picture contain {}, then show white, otherwise show black'.format(target_name)}
#             seed = generate_sample(root, img, output_img, prompt, task_type+'pos')
#             seeds.append(seed)
#             for cls in clses:
#                 if cls == cls_label:
#                     continue
#                 if random.random() > neg_sample_rate:  # 负采样率
#                     continue
#                 nname = clses[cls]
#                 output_img = get_class_img(img, nname, cls_label, is_pos=False)
#                 prompt = {'edit': 'show white if the picture has {}, otherwise show black'.format(nname)}
#                 seed = generate_sample(root, img, output_img, prompt, task_type+'neg_{}'.format(nname))
#                 seeds.append(seed)
#             n += 1
#             if n % 100 == 0:
#                 print('{} images processed!'.format(n))
#             continue
#         else:
#             output_img = get_bbox_img(root, img_id, None, dataset='oxford-pets')
#             if output_img is None:
#                 continue
#             prompt = {}
#             prompt['edit'] = 'detect the {}'.format(target_name)

#         seed = generate_sample(root, img, output_img, prompt, task_type)
#         seeds.append(seed)

#     n +=1 
#     if n % 100 == 0:
#         print('{} images processed!'.format(n))


print('begin to process coco dataset...')
#coco_fp = open(os.path.join(root, 'MSCOCO/annotations/instances_train2017.json'))

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

n = 0
for image_id in img_info:
    for cid in img_info[image_id]:
        cname = clses[cid+pet_cls_num]
        
        img_id = image_id.zfill(12)
        img_path = os.path.join(root, 'coco/train2017/{}.jpg'.format(img_id))
        img = Image.open(img_path).convert("RGB")
        #print(box, [box[0], box[1], box[0]+box[2], box[1]+box[3]])
        det_img = get_bbox_img(root, img_id, bbox = img_info[image_id][cid]['bbox'], dataset='MSCOCO')
        prompt = {'edit': 'detect the {}'.format(cname)}
        seed = generate_sample(root, img, det_img, prompt, task_type='det_{}'.format(cname))
        seeds.append(seed)

        # cls_img = get_class_img(img, cname, cid+pet_cls_num, is_pos=True)
        # prompt = {'edit': 'show white if the picture has {}, otherwise show black'.format(cname)}
        # seed = generate_sample(root, img, cls_img, prompt, task_type='cls_{}_pos'.format(cname))
        # seeds.append(seed)
        # for cls in clses:
        #     if cls == cls_label:
        #         continue
        #     if random.random() > neg_sample_rate:
        #         continue
        #     nname = clses[cls]
        #     output_img = get_class_img(img, nname, cls_label, is_pos=False)
        #     prompt = {'edit': 'show white if the picture has {}, otherwise show black'.format(nname)}
        #     seed = generate_sample(root, img, output_img, prompt, task_type='cls_{}_neg_{}'.format(cname, nname))
        #     seeds.append(seed)

        # h, w = img.size
        # gt = np.zeros((w, h), dtype=np.uint8)
        # for seg in img_info[image_id][cid]['segmentation']:
        #     for s in seg:
        #         s = np.array(s).reshape(-1, 2)     # [n_points, 2]
        #         cv2.fillPoly(gt, s.astype(np.int32)[np.newaxis, :, :], (255, 255, 255))

        # prompt = {'edit': 'segment the {}'.format(cname)}
        # seg_img = Image.fromarray(cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)).convert("RGB")
        # seed = generate_sample(root, img, seg_img, prompt, task_type='seg_{}'.format(cname))
        # seeds.append(seed)

    n +=1 
    if n % 100 == 0:
        print('{} images processed!'.format(n))

seed_file.write(json.dumps(seeds))
seed_file.close()
