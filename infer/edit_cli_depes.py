# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description **  Script for inferencing the depes task.
# ---------------------------------------------------------------------------------
# References:
# Instruct-pix2pix: https://github.com/timothybrooks/instruct-pix2pix/blob/main/edit_cli.py
# ---------------------------------------------------------------------------------

from __future__ import annotations

import math
import random
import sys
import os
import time
from argparse import ArgumentParser
from torchvision import transforms
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from evaluate.evaluate_cls_seg_det import genGT
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


def inference_depes(resolution, steps, vae_ckpt, split, test_txt_path, config, eval, single_test,
                  ckpt, input, output, edit, cfg_text, cfg_image, seed, task, rephrase):
    '''
    Modified by Yulu Gan
    6th, March, 2022
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
    
    genGT(input, output, task, test_txt_path).generate_nyuv2_gt()
    
    if single_test:
        img_list                = os.listdir(input)
        for img_name in img_list:
            
            img_id              = img_name.split(".")[0]
            img_path            = os.path.join(input, img_name)
            output_path         = os.path.join(output, img_id)
            
            input_image         = Image.open(img_path).convert("RGB")
            
            width, height       = input_image.size
            factor              = resolution / max(width, height)
            factor              = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width               = int((width * factor) // 64) * 64
            height              = int((height * factor) // 64) * 64
            input_image         = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

            prompts             = edit
            print("prompts:", prompts)
            
            if edit == "":
                input_image.save(output)
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                
                cond["c_crossattn"] = [model.get_learned_conditioning([prompts])] #modified: edit -> prompts
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

            if os.path.exists(output_path) == False:
                os.makedirs(output_path)

            edited_image.save(output_path+'/{}_depes_pred.jpg'.format(img_id))
    
    else:
        with open(test_txt_path) as file:  

            for line in file:
            
                start = time.time()
                
                img_path_part   = line.strip().split(" ")[0] # kitchen/rgb_00045.jpg
                
                file_name       = img_path_part.split("/")[0] # kitchen
                
                img_name        = img_path_part.split("/")[1] # rgb_00045.jpg

                img_id          = file_name + "_" + img_name.split(".")[0] # kitchen_rgb_00045
                
                dep_path_part = file_name + "/rgb_" + img_name.split("_")[-1] # kitchen_0028b/rgb_00045.jpg
                dep_path      = os.path.join('data/nyu_mdet', dep_path_part)
                
                input_image = Image.open(dep_path).convert("RGB")
                input_image = resize(input_image)
                
                width, height = input_image.size
                factor = resolution / max(width, height)
                factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
                width = int((width * factor) // 64) * 64
                height = int((height * factor) // 64) * 64
                input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
                
                prompts = edit
                if rephrase:
                    # prompts  = paraphrase(prompts)
                    # prompts.replace("percentage", "%")
                    prompts_pool = ['Approximate the depth of this image.',
                                    'Make an estimation of how deep the this image is.',
                                    'Provide a rough calculation of the image\'s depth.',
                                    'Give an approximate measurement of the image\'s depth.',
                                    'Make an informed guess of the depth of the image.',
                                    'Make an estimation of how deep the image goes.',
                                    'Estimate the depth of this image']
                    prompts = random.choice(prompts_pool)                
                print("prompts:", prompts)

                # if edit == "":
                #     input_image.save(output)
                #     return
                
                

                with torch.no_grad(), autocast("cuda"), model.ema_scope():
                    cond = {}
                    cond["c_crossattn"] = [model.get_learned_conditioning([prompts])]
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
                
                output_path = os.path.join(output, img_id + "_" + task)
                    
                if os.path.exists(output_path) == False:
                    os.makedirs(output_path)
                        
                edited_image.save(output_path+'/{}_{}_pred.png'.format(img_id, task))
                
                end = time.time()
                
                print("One image done. Inferenct time cost:{}".format(end - start))


if __name__ == "__main__":
    inference_depes()
