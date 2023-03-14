# Language_vison_interface

PyTorch implementation of Language_vision_interface. Largely base on  [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion) and [Instruct Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) repo. <br>

![Demo](https://i.hd-r.cn/b9dc90bd1f3895bdbff45226627b8c41.png)

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

## Prepare datasets
### NYUV2
Download the dataset [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
Or, you can download the processed dataset follow the instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/docs/dataset_prepare.md#NYU)
To get the proceesed dataset we need, you can run:
```
python dataset_creation/preprocess_nyuv2.py
```
### MS-COCO
Download the dataset [here](https://cocodataset.org/)
Then run:
```
python dataset_creation/format_dataset.py --goal generate_coco
```

### ADE20k
Download the dataset [here](http://sceneparsing.csail.mit.edu/index_challenge.html)
Then run:
```
python dataset_creation/format_dataset.py --goal generate_coco
```

### Oxford-IIIT 
Download the dataset [here](https://www.robots.ox.ac.uk/~vgg/data/pets/)
Then run:
```
python dataset_creation/format_dataset.py --goal generate_pets
```


## Generate vison figures with language
### Four tasks
```
python data/format_dataset.py
bash scripts/download_pretrained_sd.sh
```

## Train
```
python main.py --name <exp_name> --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```

## Dataset Structure
### Train


### Evaluate
```
language_vision_interface
├──scripts
├──datasets
├──data
│   ├── image_pairs_train
│   ├── image_pairs_evalation
│   │   ├── Abyssianian_1_cls
│   │   │   ├── Abyssianian_1_0
│   │   │   ├── Abyssianian_1_1
│   │   ├── Abyssianian_2_cls
│   │   │   ├── Abyssianian_2_0
│   │   │   ├── Abyssianian_2_1
│   │   ├── ...
│   │   ├── American_bulldog_100_cls
│   │   │   ├── American_bulldog_100_0
│   │   │   ├── American_bulldog_100_1
│   │   ├── ...
│   │   ├── Abyssianian_1_seg
│   │   ├── Abyssianian_2_seg
│   │   │   ├── Abyssianian_1_0
│   │   │   ├── Abyssianian_2_1
│   │   ├── ...
│   │   ├── American_bulldog_100_seg
│   │   │   ├── American_bulldog_100_0
│   │   │   ├── American_bulldog_100_1
│   │   ├── ...
│   │   ├── Abyssianian_1_det
│   │   │   ├── Abyssianian_1_0
│   │   │   ├── Abyssianian_2_1
│   │   ├── Abyssianian_2_det
│   │   │   ├── Abyssianian_1_0
│   │   │   ├── Abyssianian_2_1
│   │   ├── ...
│   │   ├── American_bulldog_100_det
│   │   │   ├── American_bulldog_100_0
│   │   │   ├── American_bulldog_100_1
│   │   ├── ...
│   │   ├── bathroom_0001_01_depes
│   │   │   ├── bathroom_0001_0
│   │   │   ├── bathroom_0001_1
│   │   ├── bathroom_0001_02_depes
│   │   │   ├── bathroom_0001_0
│   │   │   ├── bathroom_0001_1
│   │   ├── ...
│   │   ├── living_room_0010_33_depes
│   │   │   ├── living_room_0010_33_0
│   │   │   ├── living_room_0010_33_1

```

## Reproduce the Table 2.
1. Build the datasets following Sec. Prepare datasets
2. Train the model following Sec. Train
3. Evaluate by runing:
```
python evaluate/evaluate_cls_seg_det.py
python evaluate/evaluate_depth.py
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
