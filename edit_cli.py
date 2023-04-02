# Copyright (c) 2022, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Script for inferencing the four vision tasks.
# ---------------------------------------------------------------------------------

from edit_cli_cls import inference_cls
from edit_cli_depes import inference_depes
from edit_cli_det import inference_det
from edit_cli_seg import inference_seg
from edit_cli_seg_fs1000 import inference_seg_fs1000
from argparse import ArgumentParser


def main():

    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--input", required=True, type=str, help="should be the path to the file")
    parser.add_argument("--output", required=True, type=str, help="should be path to the output file")
    parser.add_argument("--edit", required=True, type=str, help="use e.g., show blue if the image has % (% is a must)")
    parser.add_argument("--cfg_text", default=7.5, type=float)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--cfg_image", default=1.5, type=float)
    parser.add_argument("--split", default="", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", default="", type=str)
    args = parser.parse_args()
    
    inference_params           = dict({"resolution": args.resolution, 
                                     "steps": args.steps, 
                                     "config": args.config,
                                     "ckpt": args.ckpt,
                                     "vae_ckpt": args.vae_ckpt,
                                     "input": args.input,
                                     "output": args.output,
                                     "edit": args.edit,
                                     "cfg_text": args.cfg_text,
                                     "cfg_image": args.cfg_image,
                                     "split": args.split,
                                     "seed": args.seed,
                                     "task": args.task,
                                     })
    
    #TODO: enable batch-level input
    
    if args.task == "seg":
        inference_seg(**inference_params)
        
    if args.task == "cls":
        inference_cls(**inference_params)
        
    if args.task == "det":
        inference_det(**inference_params)
        
    if args.task == "depes":
        inference_depes(**inference_params)
    
    if args.task == "seg_fs1000":
        inference_seg_fs1000(**inference_params)
    

if __name__ == "__main__":
    main()