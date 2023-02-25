import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import shutil
import os
import json
#from pycocotools.coco import COCO
import cv2
from revChatGPT.V1 import Chatbot

# chatbot = Chatbot(config={
#   "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJnYW55dWx1QHN0dS5wa3UuZWR1LmNuIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImdlb2lwX2NvdW50cnkiOiJVUyJ9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsidXNlcl9pZCI6InVzZXItdWNXcFN6NXg4eXpEZXJRaWRkNDl6dXk3In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMTQ1NDA2MzgxNTg3MTI3MjMxNiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzczMTI3NDgsImV4cCI6MTY3ODUyMjM0OCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.mrBudWPXnpW-s3esfdROKglho1XjmrS9OhB5TyX0Nf7J-Wf6-1tsjN3HbMJsJVMdOEqyB0z2H6TZ66_a5qkTsO07LAuwFc0scPlx1HhD9MX213M21XFW0MT2anDkCyxVV9KSAfv_cb8j0wolnkNl3oC9TJjHmbcU89YdoB7d22amLClfD1pzLPuOiOnRb_eB3qE7Eoep3eY7Smjp7hlRGKhfU_fuTbKUOwX1EXODm7aGCGU-j9BCb_9CXDAPVGMLMuvPo2Zx8z2RUgxgEq0Not86eSsJbtxZbSyLUq9grlgYyI63WMpXn7ZZLSQyTAaVi-eYpicl7meswi01DtnqtA"
# })

prompt = ["rephrase randomly: segment the cat.","rephrase randomly: segment the cat.","rephrase randomly: segment the cat.","rephrase randomly: segment the cat."]
response = ""


tasks = ['seg', 'cls', 'det']

clses = {}

def chatgpt(prompt):

    for i in range(len(prompt)):

        for data in chatbot.ask(
        prompt[i],
        ):
            response = data["message"]
        print(response)

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
        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='red', width=6)
    
    elif dataset == 'MSCOCO':
        img_path = os.path.join(root, 'coco/val2017', '%s.jpg' % img_id)

        img = Image.open(img_path).convert("RGB")
        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        for box in bbox:
            a.rectangle(box, fill=None, outline='red', width=4)

    del a

    return box_img

def get_class_img(img, target_name, cls_label):

    cls_label = int(cls_label)

    clses[cls_label] = target_name

    r = cls_label % 24
    cls_label = cls_label - r * 24
    g = cls_label % 24 if cls_label > 0 else 0
    cls_label = cls_label - g * 24
    b = cls_label % 24 if cls_label > 0 else 0

    rgb_color = (r * 10, g * 10 * 24, b * 10 * 24**2)

    #print(rgb_color)

    cls_img = Image.new('RGB', img.size, rgb_color) 
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

    img_path = os.path.join(root, 'oxford-pets/images', '%s.jpg' % img_id)
    img = Image.open(img_path).convert("RGB")

    for task_type in tasks:

        if task_type == 'seg':
            output_img = get_seg_img(root, img_id)
            prompt = {}
            prompt['edit'] = 'segment the {}'.format(target_name)

        elif task_type == 'cls':
            output_img = get_class_img(img, target_name, cls_label)
            prompt = {}
            prompt['edit'] = 'what is the animal?'.format(target_name)

        else:
            output_img = get_bbox_img(root, img_id, None, dataset='oxford-pets')
            if output_img is None:
                continue
            prompt = {}
            prompt['edit'] = 'detect the {}'.format(target_name)

        seed = generate_sample(root, img, output_img, prompt, task_type)
        seeds.append(seed)

    n +=1 
    if n % 100 == 0:
        print('{} images processed!'.format(n))

print('begin to process coco dataset...')
#coco_fp = open(os.path.join(root, 'MSCOCO/annotations/instances_train2017.json'))
coco_path = os.path.join(root, 'coco/annotations/instances_val2017.json')
coco_fp = open(coco_path)
anno_js = json.loads(coco_fp.readline())

pet_cls_num = len(clses)
for cate in anno_js['categories']:
    cid = cate['id'] + pet_cls_num
    cname = cate['name']
    clses[cid] = cname

print(clses)

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
        img_path = os.path.join(root, 'coco/val2017/{}.jpg'.format(img_id))
        img = Image.open(img_path).convert("RGB")
        #print(box, [box[0], box[1], box[0]+box[2], box[1]+box[3]])
        det_img = get_bbox_img(root, img_id, bbox = img_info[image_id][cid]['bbox'], dataset='MSCOCO')
        prompt = {'edit': 'detect the {}'.format(cname)}
        seed = generate_sample(root, img, det_img, prompt, task_type='det_{}'.format(cname))
        seeds.append(seed)

        cls_img = get_class_img(img, cname, cid+pet_cls_num)
        prompt = {'edit': 'what is the animal?'}
        seed = generate_sample(root, img, cls_img, prompt, task_type='cls_{}'.format(cname))
        seeds.append(seed)

        h, w = img.size
        gt = np.zeros((w, h), dtype=np.uint8)
        for seg in img_info[image_id][cid]['segmentation']:
            for s in seg:
                s = np.array(s).reshape(-1, 2)     # [n_points, 2]
                cv2.fillPoly(gt, s.astype(np.int32)[np.newaxis, :, :], (255, 255, 255))

        prompt = {'edit': 'segment the {}'.format(cname)}
        seg_img = Image.fromarray(cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)).convert("RGB")
        seed = generate_sample(root, img, seg_img, prompt, task_type='seg_{}'.format(cname))
        seeds.append(seed)

    n +=1 
    if n % 100 == 0:
        print('{} images processed!'.format(n))

seed_file.write(json.dumps(seeds))
seed_file.close()
