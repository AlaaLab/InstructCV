#!/bin/bash
#SBATCH -o %j.out
#SBATCH -A synth-derm
#SBATCH --gres=gpu:1
#SBATCH -p g40
#SBATCH --qos=normal
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH -t 48:00:00

# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

config=/fsx/proj-lvi/language-vision-interface/baselines/pix2seq/configs/config_multi_task.py:object_detection@coco/2017_object_detection,vit-b
model_dir=/fsx/proj-lvi/language-vision-interface/baselines/pix2seq/results
# Path to save the detected boxes for evaluating other tasks.
boxes_json_path=$model_dir/boxes.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python baselines/pix2seq/run.py --config=$config --model_dir=$model_dir --mode=eval --config.task.eval_outputs_json_path=$boxes_json_path