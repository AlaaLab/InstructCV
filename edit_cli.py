# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Script for inferencing the four vision tasks.
# ---------------------------------------------------------------------------------

from infer.edit_cli_cls import inference_cls
from infer.edit_cli_depes import inference_depes
from infer.edit_cli_det import inference_det
from infer.edit_cli_seg import inference_seg
from infer.edit_cli_seg_fs1000 import inference_seg_fs1000
from infer.edit_cli_depes_sunrgbd import inference_sunrgbd_depes
from infer.edit_cli_seg_voc import inference_seg_voc
from infer.edit_cli_det_voc import inference_det_voc
from infer.edit_cli_seg_pets import inference_seg_pets
from infer.edit_cli_seg_coco import inference_seg_coco


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
    parser.add_argument("--split", default="", type=str, help="e.g., test_part0.txt")
    parser.add_argument("--test_txt_path", default="/lustre/grp/gyqlab/lism/brt/language-vision-interface/data/nyu_mdet/nyu_test.txt", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", default="", type=str)
    parser.add_argument("--eval", action='store_true', default=False, help="Disable evaluation")
    parser.add_argument("--rephrase", action='store_true', default=False, help="Disable rephrasing prompts")
    parser.add_argument("--single_test", action='store_true', default=False, help="enable single image test")
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
                                     "rephrase": args.rephrase,
                                     "test_txt_path": args.test_txt_path,
                                     "eval": args.eval,
                                     "single_test": args.single_test
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
    
    if args.task == "fs1000_seg":
        inference_seg_fs1000(**inference_params)
        
    if args.task == "sunrgbd_depes":
        inference_sunrgbd_depes(**inference_params)
        
    if args.task == "voc_seg":
        inference_seg_voc(**inference_params)
    
    if args.task == "voc_det":
        inference_det_voc(**inference_params)
        
    if args.task == "pet_seg":
        inference_seg_pets(**inference_params)
    
    if args.task == "seg_coco":
        inference_seg_coco(**inference_params)

if __name__ == "__main__":
    main()