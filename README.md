# Language_vison_interface

PyTorch implementation of Language_vision_interface. Largely base on  [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion) and [Instruct Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) repo. <br>

## Play with your image
You can put your favorite image and try to use language as instruction to do some vision tasks. Just have a try to see what will happen!

You first need to put the image under imgs/

Then run the command bellow.

```
python edit_cli.py --input imgs/<your_img_name>.jpg --output imgs/<output_name>.jpg --edit "segment the abyssinian"
```

## Set up

```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```

## Generate vison figures with language
### Oxford-pets
```
cd data/oxford-pets
python format_dataset.py
bash scripts/download_pretrained_sd.sh
```

## Train
```
python main.py --name <exp_name> --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```


## Edit a single image:
```
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"

# Optionally, you can specify parameters to tune your result:
# python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"
```

## Baseline
### Oxford-pets
Use Resnet-50 (Pretained on ImageNet) 
After fine-tuning 100 epochs (lr=0.01, SGD), Acc(%) on test: 93.05, batch size=256
```
python baseline/classification/cls.py --model supervised --dataset pets --steps 100
```
Use ViT-16 (Pretained on ImageNet21k)
After fine-tuning 300 epochs (lr=0.001, SGD, the same as original paper), Acc(%) on test: 94.47 (94.43 in vit paper) batch size=64
```
python baseline/classification/cls.py --model ViT-16 --dataset pets --steps 300
```
