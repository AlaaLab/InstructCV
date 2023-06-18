<!-- # &#x1F309; Language vison interface -->

&#x1F31F; Official PyTorch implementation of Language Vision Interface. 

The master branch works with **PyTorch 1.5+**.

## Overview
We bulit a interface between language and vision tasks. We can use various language instructions to decide which vision task to do using one model, one loss function.

<!-- [![pCVB5B8.png](https://s1.ax1x.com/2023/06/11/pCVB5B8.png)](https://imgse.com/i/pCVB5B8)
<br/> -->

<img src="https://media.giphy.com/media/5t3Txysw5Ea2YZMrpm/giphy.gif" width="256">

<details open>
<summary>Major features</summary>

- **Diffent Vision Tasks**

    To evaluate the multi-task performance using our method, here we apply four typical vision tasks including classification, semantic segmentation, object detection and depth estimation. Our method is likely to work well on other vision tasks as well.

- **Diverse Language Instructions**

    As human usually express the same meaning with various expressions, we are working on to make diverse langanguage instructions.

</details>
<br/>


## Links
* [Project Page](https://github.com) 
* [HuggingFace 🤗 Demo](https://huggingface.co/spaces/yulu2/InstructCV)
* [ArXiv Page](https://arxiv.or)


## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@article{,
  title={{}},
  author={},
  journal={},
  year={2023}
}
```
## Set up the environments
Install dependencies by running:
```shell
#Step0. Set up the env.
conda env create -f environment.yaml
conda activate lvi
#Step1 (optional) . You could ignore this step if you do not run the baselines.
## install tensorflow : https://www.tensorflow.org/install/pip
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI 
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

## Model Zoo
<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" colspan="2">Depth <br>Estimation <br>RMSE⬇</th>
    <th align="center" style="text-align:center" colspan="2">Sementic Segmentation mIoU⬆</th>
    <th align="center" style="text-align:center" colspan="2">Classification <br>Acc⬆</th>
    <th align="center" style="text-align:center" colspan="2">Object Detection mAP⬆</th>
    <th align="center" style="text-align:center">Download</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">NYUv2</td>
    <td align="center">SUNRGB-D</td>
    <td align="center">ADE-20K</td>
    <td align="center">VOC</td>
    <td align="center">Oxford-Pets</td>
    <td align="center">ImageNet-sub</td>
    <td align="center">COCO</td>
    <td align="center">VOC</td>
  </tr>
  <tr>
    <td align="center"><a href="configs/Panoptic/odise_label_coco_50e.py"> InstructCV(rephrased) </a></td>
    <td align="center">0.297</td>
    <td align="center">0.279</td>
    <td align="center">47.235</td>
    <td align="center">52.125</td>
    <td align="center">82.135</td>
    <td align="center">74.665</td>
    <td align="center">48.500</td>
    <td align="center">61.700</td>
    <td align="center"><a href="https://github.com/"> checkpoint </a></td>
  </tr>
</tbody>
</table>

## Get Started
See [Preparing Datasets for InstructCV](datasets/README.md).

See [Getting Started with InstructCV](GETTING_STARTED.md) for detailed instructions on training and inference with ODISE.

## Play with your image
You can put your favorite image and try to use language as instruction to do some vision tasks. Just have a try to see what will happen!

**Step0.** Download the pre-trained weights we provided.
Or you can download it manually from [Google Drive](https://drive.google.com/file/d/1pz9eheQRQfx8itLj3nSKXQylTuG8DtB_/view?usp=share_link) |
[BaiduNet Disk](https://pan.baidu.com/s/1iPuMJIWTHiDBRVeFpVXUPQ?pwd=3tjr&_at_=1679742406093) 
```shell
bash scripts/download_pretain_weights.sh
```

**Step1.** Put your image under a dictionary you created.

**Step2.** Set up environments following instructions in next chapter.

**Step3.** Inference.

```shell
python edit_cli.py --input <path_to_the_dictionary_you_created> --output <path_to_save> --edit <language_instructions>
# a specific example:
python edit_cli.py --input imgs/ --output outputs/ --edit "segment the cat."
```
<br>





## Training
[Training Log](https://drive.google.com/file/d/1pMeRfWvDXSW7k7ESQBliMkgGoWQi74FW/view?usp=share_link)

### Download pre-trained models
We trained our model from the checkpoint provided by Stable Diffusion V1.5
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

After fine-tuning 100 epochs (lr=0.01, SGD), Acc(%) on test: 93.05, batch size=256
```shell
python baselines/classification/cls.py --model supervised --dataset pets --steps 100
python baselines/classification/cls.py --model supervised --dataset caltech --steps 100
```
**ViT-16 (Pretained on ImageNet21k)**

After fine-tuning 300 epochs (lr=0.001, SGD, the same as original paper), Acc(%) on test: 94.47 (94.43 in vit paper) batch size=64
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
<summary>Specialized model - Object Detection</summary>
<br>

**Generalist models**

**Unified-IO**
we use xl_1000k.bin as the pre-trained model.
It takes ~27s to inference single image.

**Pixel2Seq**
You may change all the dict[str, tf.Tensor] to dict, as it will exist error like "TypeError:  'type' is not subscriptable" caused by dependencies version differences.

Change the data_root in dataset_configs.py





</details>
<br>

## Demo

* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yulu2/InstructCV)

* Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)


The pre-trained model for Stable Diffusion is subject to its original license terms from [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

* To run InstructCV's demo from the command line:

    ```shell
    python demo/app.py --input demo/examples/coco.jpg --output demo/coco_pred.jpg --edit "Please segment the dog."
    ```
    The output is saved in `demo/coco_pred.jpg`. For more detailed options for `demo/demo.py` see [Getting Started with InstructCV](GETTING_STARTED.md).
    
  
* To run the [Gradio](https://github.com/gradio-app/gradio) demo locally:
    ```shell
    python demo/app.py
    ```


## Acknowledgement

Code is largely based on [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion) and [Instruct Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix).

Thank you, all, for the great open-source projects!