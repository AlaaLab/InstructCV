## InstructCV: Instruction-Tuned Text-to-Image Diffusion Models as Vision Generalists

Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis and Ahmed Alaa

[Paper](https://arxiv.or) | [HuggingFace 🤗 Demo](https://huggingface.co/spaces/yulu2/InstructCV)

&#x1F31F; Official PyTorch implementation of **InstructCV**. The master branch works with **PyTorch 1.5+**.

![Screenshot 2023-09-30 at 11 30 35 AM](https://github.com/AlaaLab/InstructCV/assets/21158134/0baadd7a-7b43-4766-a3fb-0c9f6ba9f65f)


## Overview
Recent advances in generative diffusion models have enabled text-controlled synthesis of realistic and diverse images with impressive quality. Despite these remarkable advances, the application of text-to-image generative models in computer vision for standard visual recognition tasks remains limited. The current de facto approach for these tasks is to design model architectures and loss functions that are tailored to the task at hand. In this project, we develop a unified language interface for computer vision tasks that abstracts away task specific design choices and enables task execution by following natural language instructions. Our approach involves casting multiple computer vision tasks as text-to-image generation problems. Here, the text represents an instruction describing the task, and the resulting image is a visually-encoded task output. To train our model, we pool commonly-used computer vision datasets covering a range of tasks, including segmentation, object detection, depth estimation, and classification. We then use a large language model to paraphrase prompt templates that convey the specific tasks to be conducted on each image, and through this process, we create a multi-modal and multi-task training dataset comprising input and output images along with annotated instructions. Following the InstructPix2Pix architecture, we apply instruction-tuning to a text-to-image diffusion model using our constructed dataset, steering its functionality from a generative model to an instruction-guided multi-task vision learner. 

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

## Results
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
    <td align="center"><a href="configs/Panoptic/odise_label_coco_50e.py"> InstructCV-RP </a></td>
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
