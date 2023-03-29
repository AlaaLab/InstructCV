# for ((i=0;i<=7;i++))
# do
# (
#   echo "$i"
#   CUDA_VISIBEL_DEVICES=i python edit_cli_det.py --split "test_part$i.txt" --ckpt logs/train_all100kdata_new/checkpoints/epoch=000051.ckpt --input data/coco/ --output outputs/imgs_test_coco/ --edit "detect the %" --task det
# ) &
# done
# wait
# echo -E "########## $SECONDS ##########"

CUDA_VISIBEL_DEVICES=4 python edit_cli_det.py --split "test_part6.txt" --ckpt logs/train_all100kdata_new/checkpoints/epoch=000051.ckpt --input data/coco/ --output outputs/imgs_test_coco/ --edit "detect the %" --task det
