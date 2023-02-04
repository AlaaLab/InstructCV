#!/bin/sh


python universal_mae.py --batch_size 128 --epochs 50 --model large --lr 0.01 --finetune False --dataset_name oxford-pets --task segmentation

#python universal_mae.py --batch_size 128 --epochs 50 --model large --lr 0.000015 --finetune True --dataset_name voc --task segmentation

#python universal_mae.py --batch_size 128 --epochs 5 --model large --lr 0.001 --finetune True #--dataset_name maze maze --task path_finding_5x5 --task_list path_finding_5x5 path_finding_6x6 --multi_task True

#python universal_mae.py --batch_size 128 --epochs 50 --model large --lr 0.00015 --finetune True
# python universal_mae.py --batch_size 64 --epochs 50 --model large --lr 0.00015 --finetune True
# python universal_MAE_pretrain.py --batch_size 64 --epochs 50 --model large --lr 0.000015 --finetune True



#set -o errexit

#EPOCHS=50
#LR=0.00015
#BATCH_SIZE=128
#FINETUNE=True
#MODEL_TYPE=large
#DATASET=maze
#COMET_LOG=True

#for task_id in path_finding_5x5 path_finding_6x6 path_finding_7x7 path_finding_9x9 path_finding_10x10

#do

#   echo "Running experiment for task $task_id on dataset $DATASET"
#   python universal_mae.py --dataset_name $DATASET --task $task_id --batch_size $BATCH_SIZE --epochs $EPOCHS --model $MODEL_TYPE --lr $LR --finetune $FINETUNE --enable_comet_logger $COMET_LOG
       
#done


