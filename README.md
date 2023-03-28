<!-- # &#x1F309; Language vison interface -->

&#x1F31F; Official PyTorch implementation of Language Vision Interface. 

The master branch works with **PyTorch 1.5+**.

## Overview
We bulit a interface between language and vision tasks. We can use various language instructions to decide which vision task to do!

![Demo](https://i.hd-r.cn/b9dc90bd1f3895bdbff45226627b8c41.png)
<br/>

<details open>
<summary>Major features</summary>

- **Diffent Vision Tasks**

    To evaluate the multi-task performance using our method, here we apply four typical vision tasks including classification, semantic segmentation, object detection and depth estimation. Our method is likely to work well on other vision tasks as well.

- **Diverse Language Instructions**

    As human usually express the same meaning with various expressions, we are working on to make diverse langanguage instructions.

</details>
<br/>

# &#x1F4A1; Play with your image
You can put your favorite image and try to use language as instruction to do some vision tasks. Just have a try to see what will happen!

**Step0.** Download the pre-trained weights we provided.
Or you can download it manually from [Google Drive](https://drive.google.com/file/d/1pz9eheQRQfx8itLj3nSKXQylTuG8DtB_/view?usp=share_link) |
[BaiduNet Disk](https://pan.baidu.com/s/1iPuMJIWTHiDBRVeFpVXUPQ?pwd=3tjr&_at_=1679742406093) 
```shell
bash download_pretain_weights.sh
```

**Step1.** Put your image under a dictionary you created.

**Step2.** Set up environments following instructions in next chapter.

**Step3.** Inference.

```shell
python edit_cli.py --input <path_to_the_dictionary_you_created> --output <path_to_save> --edit <language_instructions>
# a specific example:
python edit_cli.py --input imgs/ --output outputs/ --edit "segment the cat."
```

# &#x1F449; Training
[Training Log](https://drive.google.com/file/d/1pMeRfWvDXSW7k7ESQBliMkgGoWQi74FW/view?usp=share_link)
## Set up the environments

```shell
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```

## Prepare datasets
We pool all four datasets together and train them at one time.
<details open>
<summary>NYUV2 - Depth estimation</summary>

Download the dataset [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

Or, you can download the processed dataset follow the instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/docs/dataset_prepare.md#NYU).
</details>

<details open>
<summary>MS-COCO - Object Detection</summary>

Download the dataset [here](https://cocodataset.org/)
</details>

<details open>
<summary>ADE20k - Semantic Segmentation</summary>

Download the dataset [here](http://sceneparsing.csail.mit.edu/index_challenge.html)
</details>

<details open>
<summary>Oxford-IIIT - Classification</summary>

Download the dataset [here](https://www.robots.ox.ac.uk/~vgg/data/pets/)
</details>
<br/>

```shell
python dataset_creation/format_dataset.py --save_root <path_to_save> --tasks <vision tasks> --data_root <path_to_dataset>
# specific examples
## coco
python dataset_creation/format_dataset.py --save_root './image_pairs' --tasks ['det'] --data_root './data/coco'
```

## Train with multi-gpus

```shell
python main.py --name <exp_name> --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```

# &#x2714; Dataset Structure
## Train


## Evaluate
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

# &#x2B50; Repoduce the results in Table 1.

## Semantic segmantation
We evaluate model's performance on ADE20k



# &#x1F3B7; Baseline

<details open>
<summary>Oxford-pets</summary>
<br>

**Resnet-50 (Pretained on ImageNet)**

After fine-tuning 100 epochs (lr=0.01, SGD), Acc(%) on test: 93.05, batch size=256
```shell
python baseline/classification/cls.py --model supervised --dataset pets --steps 100
```
**ViT-16 (Pretained on ImageNet21k)**

After fine-tuning 300 epochs (lr=0.001, SGD, the same as original paper), Acc(%) on test: 94.47 (94.43 in vit paper) batch size=64
```shell
python baseline/classification/cls.py --model ViT-16 --dataset pets --steps 300
```
</details>

<details open>
<summary>ADE20k</summary>
<br>

**SegFormer**

download the pretrained weights (SegFormer-B5) from [here](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing).
```shell
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file
```

**Mask2Former**

download the pretrained weights (Swin-L IN2k with 160k iterations) from [here](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)

</details>
<br>

<details open>
<summary>NYUv2</summary>
<br>

**BTS**

We follow instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/bts) to reproduce the results.

**Binsformer**

We follow instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/binsformer) to reproduce the results.
</details>
<br>

## Acknowledgement
This project is based on the following open-source projects. We thank their authors for making the source code publically available.
* [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion)
* [Instruct Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) (largely based on this repo)