&#x1F31F; Instructions for training. 

## Inference with new images

You can apply InstructCV to new images by following the steps below.

**Step 1.** Download the pre-trained weights we provided.
Or you can download it manually from [Google Drive](https://drive.google.com/file/d/1pz9eheQRQfx8itLj3nSKXQylTuG8DtB_/view?usp=share_link) |
[BaiduNet Disk](https://pan.baidu.com/s/1iPuMJIWTHiDBRVeFpVXUPQ?pwd=3tjr&_at_=1679742406093) 
```shell
bash scripts/download_pretain_weights.sh
```

**Step 2.** Run the following command:

```shell
python edit_cli.py --input <path_to_the_dictionary_you_created> --output <path_to_save> --edit <language_instructions>
# a specific example:
python edit_cli.py --input imgs/ --output outputs/ --edit "segment the cat."
```
<br>





## Training
[Training Log](https://drive.google.com/file/d/1pMeRfWvDXSW7k7ESQBliMkgGoWQi74FW/view?usp=share_link)

### Download pre-trained models
We trained our model using the checkpoint provided by Stable Diffusion V1.5
```shell
#  Stable Diffusion V1.5
bash scripts/download_checkpoints.sh
#  The checkpoint we provided (finetune with our training data for 50 epochs)
bash scripts/download_pretrained_weights.sh
```


### Train with multi-gpus

```shell
python main.py --name <exp_name> --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```

### Train on slurm clusters
```shell
sbatch scripts/slurm_train
```


## Baseline

<details open>
<summary>Specialized model - Classification</summary>
<br>

**Resnet-50 (Pretained on ImageNet)**

```shell
python baselines/classification/cls.py --model supervised --dataset pets --steps 100
python baselines/classification/cls.py --model supervised --dataset caltech --steps 100
```
**ViT-16 (Pretained on ImageNet21k)**

```shell
python baselines/classification/cls.py --model ViT-16 --dataset pets --steps 300
```

</details>
<br>


<details open>
<summary>Specialized model - Semantic Segmentation</summary>
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
<summary>Specialized model - Monocular Depth Estimation</summary>
<br>

**BTS**

We follow instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/bts) to reproduce the results.

**Binsformer**

We follow instructions [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/binsformer) to reproduce the results.
</details>
<br>


<details open>
<summary>Specialized model - Object Detection</summary>
<br>

**Faster RCNN**
We run Faster R-CNN models
in [Detectron2](https://github.com/facebookresearch/detectron2)

**Mask RCNN**
We run Mask R-CNN models (Backbone: R-101-FPN, Lr schd: 2x)
in [mmdetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)


**DETR**
We follow instructions [here](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr) to reproduce the results.

</details>
<br>

<details open>
<summary>Vision generalists</summary>
<br>

**Generalist models**

**Unified-IO**
we use xl_1000k.bin as the pre-trained model.
It takes ~27s to inference single image.
**Pixel2Seq**
To repoduce their results using [repo](https://github.com/google-research/pix2seq) they provided, you need to change all the dict[str, tf.Tensor] to dict, as it will exist error like "TypeError:  'type' is not subscriptable" caused by dependencies version differences.

Change the data_root in dataset_configs.py





</details>
<br>
