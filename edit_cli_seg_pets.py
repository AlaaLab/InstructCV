from __future__ import annotations

import math
import random
import sys
import os
import pdb
import time
from fnmatch import fnmatch
from torchvision import transforms
from argparse import ArgumentParser

import einops
import shutil
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from dataset_creation.format_dataset import get_seg_img

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

Pet_CLASSES = ('Abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle','Bengal',#6
               'Birman', 'Bombay', 'boxer', 'British Shorthair', 'chihuahua', 'Egyptian Mau', 'english cocker spaniel',#13
               'english setter', 'german shorthaired', 'great pyrenees', 'havanese', 'japanese chin',#18
               'keeshond', 'leonberger', 'Maine Coon', 'miniature pinscher', 'newfoundland', 'Persian',#24
               'pomeranian', 'pug', 'Ragdoll', 'Russian Blue', 'saint bernard', 'samoyed', 'scottish terrier',#31
               'shiba inu', 'Siamese', 'Sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier')#37


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

def generate_pets_gt(input, output):
    
    print("generating the gt.")
    
    clses = {}
    seeds = []
    
    for line in open(os.path.join(input, 'annotations/test.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[1]

        cname = ' '.join(img_id.split('_')[:-1]).strip()
        clses[cls_label] = cname

    for line in open(os.path.join(input, 'annotations/test.txt')):
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        cls_label = words[1]

        cname = ' '.join(img_id.split('_')[:-1]).strip()
        
        output_img = get_seg_img(input, img_id)
        ge_path             = os.path.join(output, img_id)
        
        if os.path.exists(ge_path)  == False:
            os.makedirs(ge_path)
            
        gt_save_path        = os.path.join(ge_path, img_id + "_gt.png")
        output_img.save(gt_save_path)
        
    print("done with generating the g.t.")

    return 



def inference_seg_pets(resolution, steps, vae_ckpt, split, config, eval, test_txt_path, single_test,
                  ckpt, input, output, edit, cfg_text, cfg_image, seed, task, rephrase):
    '''
    Modified by Yulu Gan
    March 31, 2022
    1. Support multiple images inference
    2. Make outputs' size are the closest as inputs
    '''

    resize = transforms.Resize((resolution,resolution))
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if seed is None else seed
    generate_pets_gt(input,output)
    
    for line in open(os.path.join(input,'annotations/test.txt')):
        
        start               = time.time()
        
        line = line.strip()
        words = line.split(' ')
        img_id = words[0]
        img_name =  img_id + ".jpg"
        img_path = os.path.join(input, 'images', img_name)
        
        ge_path             = os.path.join(output, img_id)
        pred_save_path      = os.path.join(ge_path, img_id + "_pred.png")
        cname               = ' '.join(img_id.split('_')[:-1]).strip()
        print("cname:", cname)

        prompts             = edit.replace("%", cname)
        
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            
            input_image = Image.open(img_path).convert("RGB")
            input_image = resize(input_image)

            width, height = input_image.size
            factor = resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
            
            cond = {}
            
            cond["c_crossattn"] = [model.get_learned_conditioning([prompts])] #modified: args.edit -> prompts
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            
        edited_image.save(pred_save_path)
        
        end = time.time()
        
        print("One image done. Inferenct time cost:{}".format(end - start))


if __name__ == "__main__":
    inference_seg_pets()