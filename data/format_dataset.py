import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import shutil
import os
import json
import pdb
#from pycocotools.coco import COCO
import cv2
import random
import copy
from revChatGPT.V1 import Chatbot

def get_bbox_prompt(cname):
    
    prompts = {}
    flag = random.randint(1,len(det_prompts)-1)
    prompt = det_prompts[flag]
    prompt = copy.deepcopy(prompt)
    prompt = prompt.replace("%", cname)
    prompts['edit'] = prompt

    return prompts

def get_seg_prompt(cname):

    prompts = {}
    flag = random.randint(1,len(seg_prompts)-1)
    prompt = seg_prompts[flag]
    prompt = copy.deepcopy(prompt)
    prompt = prompt.replace("%", cname)
    prompts['edit'] = prompt
    return prompts

def get_cls_prompt(c, cname):
    
    num = random.randint(1,21)
    if num == 1:
        prompt = {'edit': 'Show {} if the picture has {}, otherwise show black'.format(c, cname)}
    if num == 2:
        prompt = {'edit': 'If the image has a {}, show it in {}, otherwise show black'.format(cname, c)}
    if num == 3:
        prompt = {'edit': 'If the picture shows a {}, display it in {}, otherwise show black.'.format(cname, c)}      
    if num == 4:
        prompt = {'edit': 'Display {} if the image contains a {}, otherwise show black.'.format(c, cname)}
    if num == 5:
        prompt = {'edit': 'In case the image depicts a {}, show it in {}, else display black.'.format(cname,c)}
    if num == 6:
        prompt = {'edit': 'Show the image in {} if it contains a {}, otherwise display black'.format(c, cname)}
    if num == 7:
        prompt = {'edit': 'Display {} when the image depicts a {}, otherwise show black'.format(c, cname)}
    if num == 8:
        prompt = {'edit': 'Show {} color if the image contains a {}, otherwise show black color'.format(c, cname)}
    if num == 9:
        prompt = {'edit': 'If the image includes a {}, display it in {} hue, otherwise show black'.format(cname, c)}
    if num == 10:
        prompt = {'edit': 'Display {} tint if the picture has a {}, otherwise show a black tint'.format(c, cname)}
    if num == 11:
        prompt = {'edit': 'In case the image shows a {}, show it in {} tone, otherwise show black'.format(cname, c)}
    if num == 12:
        prompt = {'edit': 'Show {} shades if the image features a {}, otherwise display black.'.format(c, cname)}
    if num == 13:
        prompt = {'edit': 'Display {} coloring if the picture shows a {}, otherwise show black.'.format(c, cname)}
    if num == 14:
        prompt = {'edit': 'In the case of a {} being present in the image, display it in {}, otherwise show black'.format(cname, c)}
    if num == 15:
        prompt = {'edit': 'If there is a {} in the picture, show it in {} hue, otherwise show black'.format(cname, c)}
    if num == 16:
        prompt = {'edit': 'When the image contains a {}, display it in {} color, otherwise show black'.format(cname, c)}
    if num == 17:
        prompt = {'edit': 'Show {} tones if the image contains a {}, otherwise show black'.format(c, cname)}
    if num == 18:
        prompt = {'edit': 'Display the image in {} if it shows a {}, otherwise show black'.format(c, cname)}
    if num == 19:
        prompt = {'edit': 'If there is a {} present in the image, display it in {}, otherwise show black'.format(cname, c)}
    if num == 20:
        prompt = {'edit': 'In case the picture features a {}, display it in {} shade, otherwise show black'.format(cname, c)}
    if num == 21:
        prompt = {'edit': 'When the image depicts a {}, display it in {} hue, otherwise show black'.format(cname, c)}
        
    return prompt

def get_depth_prompt():
    
    prompts = {}
    flag = random.randint(1,len(dep_est_prompts)-1)
    prompt = dep_est_prompts[flag]
    prompts['edit'] = prompt

    return prompts

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

def get_class_img(img, target_name, cls_label, color, is_pos):
    if is_pos:
        cls_img = Image.new('RGB', img.size, color)
    else:
        cls_img = Image.new('RGB', img.size, (0,0,0))
    return cls_img

def get_depth_img(root, img_id):
    
    img_path = os.path.join(root, 'nyuv2/depths','%s.png' % img_id)
    depth_img = Image.open(img_path).convert("RGB")
    
    return depth_img

def generate_sample(root, img, img_id, out_img, prompt, task_type):
    '''
    Args img: input/original images
         out_img: add bbox for det; mask for seg; depth img for DE .. 
         prompt: language instructions
         
    Return seed
    '''

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

def proc_oxford_pets(oxford_pets_root, tasks):

    n = 0
    seeds = []
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
                output_img = get_seg_img(root, img_id)
                prompt = {}
                prompt['edit'] = 'segment the {}'.format(target_name)

            elif task_type == 'cls':
                c = random.choice(lcolor)
                color = colors[c]
                output_img = get_class_img(img, target_name, cls_label, color, is_pos=True)
                prompt = {'edit': 'show {} if the picture contain {}, otherwise show black'.format(c, target_name)}

                seed = generate_sample(root, img, img_id, output_img, prompt, task_type+'pos')
                seeds.append(seed)
                for cls in clses:
                    if cls == cls_label:
                        continue
                    if random.random() > neg_sample_rate:  # 负采样率
                        continue
                    nname = clses[cls]
                    c = random.choice(lcolor)
                    color = colors[c]
                    output_img = get_class_img(img, nname, cls_label, color, is_pos=False)
                    prompt = {'edit': 'show {} if the picture has {}, otherwise show black'.format(c, nname)}
                    seed = generate_sample(root, img, img_id, output_img, prompt, task_type+'neg_{}'.format(nname))
                    seeds.append(seed)
                n += 1
                if n % 100 == 0:
                    print('{} images processed!'.format(n))
                continue
            else:
                output_img = get_bbox_img(root, img_id, None, dataset='oxford-pets')
                if output_img is None:
                    continue
                prompt = {}
                prompt['edit'] = 'detect the {}'.format(target_name)
            seed = generate_sample(root, img, img_id, output_img, prompt, task_type)
            seeds.append(seed)

        n +=1 
        if n % 100 == 0:
            print('{} images processed!'.format(n))
        
    return seeds

def preproc_coco(clses):
    
    print('begin to pre-process coco dataset...')
    
    pet_cls_num = len(clses)
    coco_path = os.path.join(root, coco_root, 'annotations/instances_val2017.json')
    coco_fp = open(coco_path)
    anno_js = json.loads(coco_fp.readline())

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
    
    return img_info, pet_cls_num

def proc_coco(clses, tasks):
    
    print('begin to process coco dataset...')
    
    seeds = []
    n = 0
    img_info, pet_cls_num = preproc_coco(clses)
    
    for image_id in img_info:
        for cid in img_info[image_id]:
            cname = clses[cid+pet_cls_num]
            
            img_id = image_id.zfill(12)
            img_path = os.path.join(root, 'coco/val2017/{}.jpg'.format(img_id))
            img = Image.open(img_path).convert("RGB")
            #print(box, [box[0], box[1], box[0]+box[2], box[1]+box[3]])
            det_img = get_bbox_img(root, img_id, bbox = img_info[image_id][cid]['bbox'], dataset='MSCOCO')
            
            prompt  = get_bbox_prompt(cname)
            # prompt = {'edit': 'detect the {}'.format(cname)}
                
            seed = generate_sample(root, img, img_id, det_img, prompt, task_type='det_{}'.format(cname))
            seeds.append(seed)

            c = random.choice(lcolor)
            color = colors[c]
            cls_img = get_class_img(img, cname, cid+pet_cls_num, color, is_pos=True)
            
            # prompt = {'edit': 'show {} if the picture has {}, otherwise show black'.format(c, cname)}
            prompt  = get_cls_prompt(c, cname)
            
            seed = generate_sample(root, img, img_id, cls_img, prompt, task_type='cls_{}_pos'.format(cname))
            seeds.append(seed)
            for cls in clses:
                if cls == cls_label:
                    continue
                if random.random() > neg_sample_rate:
                    continue
                nname = clses[cls]
                c = random.choice(lcolor)
                color = colors[c]
                output_img = get_class_img(img, nname, cls_label, color, is_pos=False)
                
                # prompt = {'edit': 'show {} if the picture has {}, otherwise show black'.format(c, nname)}
                prompt  = get_cls_prompt(c, nname)
                
                seed = generate_sample(root, img, img_id, output_img, prompt, task_type='cls_{}_neg_{}'.format(cname, nname))
                seeds.append(seed)

            h, w = img.size
            gt = np.zeros((w, h), dtype=np.uint8)
            for seg in img_info[image_id][cid]['segmentation']:
                for s in seg:
                    s = np.array(s).reshape(-1, 2)     # [n_points, 2]
                    cv2.fillPoly(gt, s.astype(np.int32)[np.newaxis, :, :], (255, 255, 255))

            # prompt = {'edit': 'segment the {}'.format(cname)}
            prompt  = get_seg_prompt(cname)
            
            seg_img = Image.fromarray(cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)).convert("RGB")
            seed = generate_sample(root, img, img_id, seg_img, prompt, task_type='seg_{}'.format(cname))
            seeds.append(seed)

        n +=1 
        if n % 100 == 0:
            print('{} images processed!'.format(n))
        
    return seeds

def proc_nyuv2(nyuv2_root):

    print('begin to process NYU_V2 dataset...')
    
    seeds = []
    prompt = {}
    n = 0
    
    imgs = os.listdir(os.path.join(nyuv2_root, 'images/'))
    
    for img_n in imgs:
        
        line  = img_n.strip()
        word  = line.split('.')
        img_id = word[0]
    
        img_path = os.path.join(nyuv2_root, 'images', '%s.jpg' % img_id)
        img = Image.open(img_path).convert("RGB")
        depth_img = get_depth_img(root, img_id)
        
        # prompt['edit'] = 'Estimate the depth of this image'
        prompt = get_depth_prompt()
        
        seed = generate_sample(root, img, img_id, depth_img, prompt, task_type="depes")
        seeds.append(seed)
    
    return seeds

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
    
    
    
    
    return
  
def chat():
    
    print("111")
    prompt, prompt_loc = prompts_chat()
    print("222")
    chatbot = Chatbot(config={
    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJnYW55dWx1QHN0dS5wa3UuZWR1LmNuIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImdlb2lwX2NvdW50cnkiOiJVUyJ9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsidXNlcl9pZCI6InVzZXItdWNXcFN6NXg4eXpEZXJRaWRkNDl6dXk3In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMTQ1NDA2MzgxNTg3MTI3MjMxNiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2NzczMTI3NDgsImV4cCI6MTY3ODUyMjM0OCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.mrBudWPXnpW-s3esfdROKglho1XjmrS9OhB5TyX0Nf7J-Wf6-1tsjN3HbMJsJVMdOEqyB0z2H6TZ66_a5qkTsO07LAuwFc0scPlx1HhD9MX213M21XFW0MT2anDkCyxVV9KSAfv_cb8j0wolnkNl3oC9TJjHmbcU89YdoB7d22amLClfD1pzLPuOiOnRb_eB3qE7Eoep3eY7Smjp7hlRGKhfU_fuTbKUOwX1EXODm7aGCGU-j9BCb_9CXDAPVGMLMuvPo2Zx8z2RUgxgEq0Not86eSsJbtxZbSyLUq9grlgYyI63WMpXn7ZZLSQyTAaVi-eYpicl7meswi01DtnqtA"
    })
    print("333")
    for i in range(len(prompt)):
        prompt_ask = "rephrase randomly: " + prompt[i] + '.'
        for data in chatbot.ask(
            prompt_ask,
    ):
            response = data["message"]
            
        print(response)
    
    return


if __name__ == "__main__":
    
    tasks = ['seg', 'cls', 'det']

    neg_sample_rate=0.005 # negtive sample rate

    # colors = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
    #           'white':(255,255,255), 'brown':(165,42,42), 'orange':(255,165,0),
    #           'purple':(128,0,128)}
    
    colors = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 'purple':(128,0,128)}
    
    lcolor = list(colors.keys())

    clses = {}
    
    root = ''
    oxford_pets_root = 'oxford-pets'
    coco_root = 'coco'
    nyuv2_root = 'nyuv2'

    sample_path =  os.path.join(root, 'image_pairs')

    if os.path.exists(sample_path) == False:
        os.mkdir(sample_path)

    seed_file = open(os.path.join(root, 'image_pairs', 'seeds.json'), 'w')
    
    
    #generate prompts dict
    num_seg, num_det, num_dep_est = 0, 0, 0
    seg_prompts, det_prompts, dep_est_prompts = {}, {}, {}
    
    with open("seg_prompts.txt") as file:
        for item_seg in file:
            sen_seg = item_seg.strip()
            seg_prompts[num_seg] = sen_seg
            num_seg += 1
    
    with open("det_prompts.txt") as file:
        for item_det in file:
            sen_det = item_det.strip()
            det_prompts[num_det] = sen_det
            num_det += 1
    
    with open("dep_est_prompts.txt") as file:
        for item_dep_est in file:
            sen_dep_est = item_dep_est.strip()
            dep_est_prompts[num_dep_est] = sen_dep_est
            num_dep_est += 1

    for line in open(os.path.join(root, oxford_pets_root, 'annotations/trainval.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[1]

        target_name = ' '.join(img_id.split('_')[:-1]).strip()
        clses[cls_label] = target_name #store target_name and cls_label
    
    
    seeds_coco = proc_coco(clses, tasks)
    # seeds_pets = proc_oxford_pets(oxford_pets_root, tasks)
    seeds_nyuv2 = proc_nyuv2(nyuv2_root)
    
    seeds = seeds_nyuv2 + seeds_coco
    
    seed_file.write(json.dumps(seeds))
    seed_file.close()
