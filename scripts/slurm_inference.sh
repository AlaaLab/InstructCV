#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu32
#SBATCH --qos=normal
#SBATCH -J test
#SBATCH --nodes=1 
#SBATCH -t 48:00:00


# for pet seg
CUDA_VISIBLE_DEVICES=0 python edit_cli.py --resolution 256 --ckpt logs/train_all100kdata_add_coco_pet_seg_blue/checkpoints/epoch=000020.ckpt --input data/oxford-pets --output ./outputs/imgs_test_pets_seg/ --edit "segment the %" --task pet_seg