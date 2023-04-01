from __future__ import annotations

import math
import random
import sys
import os
import pdb
import time
import json
from torchvision import transforms
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def preproc_coco(root):
    
    print('begin to pre-process coco dataset...')
    clses                   = {}
    coco_path               = os.path.join(root, 'annotations/instances_val2017.json')
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

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def inference_det():
    '''
    Modified by Yulu Gan
    6th, March, 2022
    1. Support multiple images inference
    2. Make outputs' size are the closest as inputs
    '''
    
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str, help="should be the path to the file")
    parser.add_argument("--output", required=True, type=str, help="should be path to the output file")
    parser.add_argument("--edit", required=True, type=str, help="use e.g., show blue if the image has % (% is a must)")
    parser.add_argument("--split", default="", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", default="", type=str)
    args = parser.parse_args()

    resize = transforms.Resize([512,512])
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    
    for image_name in open(os.path.join(args.input,args.split)): #"test_part0.txt"
        
        start                   = time.time()
        
        image_name              = image_name.strip()
        
        img_info, clses         = preproc_coco(args.input)
        
        image_id                = image_name.split(".")[0] #000001234
        img_id                  = image_id.lstrip("0") #1234
            
        img_path = os.path.join(args.input, 'val2017/{}.jpg'.format(image_id))
        
        if img_id not in img_info:
            
            continue

        for cid in img_info[img_id]:
            
            cname               = clses[cid] #target_name
            output_path         = os.path.join(args.output, image_id + "_" + cname + "_" + args.task)
            
            # # resume
            # if os.path.exists(output_path) == True:
            #     continue

            input_image         = Image.open(img_path).convert("RGB")
            input_image         = resize(input_image)
            
            width, height       = input_image.size
            factor              = args.resolution / max(width, height)
            factor              = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width               = int((width * factor) // 64) * 64
            height              = int((height * factor) // 64) * 64
            input_image         = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

            prompts             = args.edit
            prompts             = prompts.replace("%", cname)
            print("prompts:", prompts)
            
            if args.edit == "":
                input_image.save(args.output)
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                
                cond["c_crossattn"] = [model.get_learned_conditioning([prompts])] #modified: args.edit -> prompts
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            

            if os.path.exists(output_path) == False:
                os.makedirs(output_path)

            edited_image.save(output_path+'/{}_{}_pred.jpg'.format(image_id + "_" + cname, args.task))

            end = time.time()
            
            print("One image done. Inferenct time cost:{}".format(end - start))


if __name__ == "__main__":
    inference_det()
