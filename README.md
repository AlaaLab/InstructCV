## InstructCV: Instruction-Tuned Text-to-Image Diffusion Models as Vision Generalists

Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis, Ahmed Alaa

&#x1F31F; Official PyTorch implementation of Language Vision Interface. 

The master branch works with **PyTorch 1.5+**.

## Overview
We bulit a interface between language and vision tasks. We can use various language instructions to decide which vision task to do using one model, one loss function.

<!-- [![pCVB5B8.png](https://s1.ax1x.com/2023/06/11/pCVB5B8.png)](https://imgse.com/i/pCVB5B8)
<br/> -->

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2xneXN0czhtMXFxaTJuNThrb3NyMGRkcGQwaDAwaHd6c3h2cHlxNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Rs9tOqqKCqqfoAKjsZ/giphy.gif" width="90%">
</div>
<br>
<div align="center">
<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjdwcnFqOGI1MndqZGV5MTgxd2NldjR2YWp4dzFicnF2Y3c3bnFkbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MIfEWTuLeMQNAsKOas/giphy.gif" width="44.2%">
<img src="https://media.giphy.com/media/5t3Txysw5Ea2YZMrpm/giphy.gif" width="44.5%">
</div>
<br>
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjZmNTBhOTY2NmQ3ZDNhMTIyNTI3ZGQ5MDIzNjRmNzE0YzJhNmE4MSZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/XzWdZ1KZCRfgQMZnW6/giphy.gif" width="44.5%">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjFma2FndWlhZG01dm0ybzFncTg5cWZuNXc1NGd6bDUwYmdzbzNrYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/B8qVvI0HhEkVdTCBLe/giphy.gif" width="44.3%">
</div>

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
See [Preparing Datasets for InstructCV](DATASET.md).

See [Getting Started with InstructCV](GETTING_STARTED.md) for detailed instructions on training and inference with InstructCV.


## Demo

* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/alaa-lab/InstructCV)

* Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YDI2kb6uPP1d1VsiarFDapufRtkYso4g)


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
