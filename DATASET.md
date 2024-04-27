&#x1F31F; Instructions for generating dataset we proposed. 


# Prepare training datasets
## Structure
### Train

```
language_vision_interface
├──scripts
├──data
│   ├── image_pairs_train
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
│   │   │   ├── Abyssianian_1_0
│   │   │   ├── Abyssianian_1_1
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
│   │   │   ├── Abyssianian_1_1
│   │   ├── Abyssianian_2_det
│   │   │   ├── Abyssianian_2_0
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
<br>

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

Download the dataset [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
Download the instance annotation from [here](http://sceneparsing.csail.mit.edu/)
```
cd ADEChallengeData2016
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
```
</details>

<details open>
<summary>Oxford-IIIT - Classification</summary>

Download the dataset [here](https://www.robots.ox.ac.uk/~vgg/data/pets/)
</details>
<br/>

External dataset for testing:
<details open>
<summary>SUNRGBD - Depth estimation</summary>

Download the dataset [here](https://rgbd.cs.princeton.edu/) and download the split file from this [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/splits). We remove NYUv2 part.

</details>


<details open>
<summary>PASCAL VOC2012 - Segmentation & Detection</summary>

Download the dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

We need to transfer the voc format to the coco one by running:
```shell
python data/VOCdevkit/VOC2012/voc2coco.py
```
</details>

## Build our training data

Next, we are going to process these datasets to build our training data. You can run the following commands.

```shell
python dataset_creation/format_dataset.py --save_root <path_to_save> --tasks <vision tasks> --data_root <path_to_dataset>
# specific examples
## coco
python build_data/format_dataset_rp.py --save_root './image_pairs' --tasks ['det'] --data_root './data/coco'
```