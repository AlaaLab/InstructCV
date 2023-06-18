# ------------------------------------------------------------------------------
# Copyright (c) 2023, Alaa lab, UC Berkeley. All rights reserved.
#
# Written by Yulu Gan
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
import pdb
import cv2
import random
from fnmatch import fnmatch
import numpy as np

import gradio as gr
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline

title = "InstructCV"

description = """
<p style='text-align: center'> <a href='https://huggingface.co/spaces/yulu2/InstructCV/' target='_blank'>Project Page</a> | <a href='https://arxiv.org' target='_blank'>Paper</a> | <a href='https://github.com' target='_blank'>Code</a></p>
Gradio demo for InstructCV: Towards Universal Text-to-Image Vision Generalists. \n
You may upload any images you like and try to let the model do vision tasks following your intent. \n
Some examples: You could use "Segment the dog" for segmentation, "Detect the dog" for object detection, "Estimate the depth map of this image" for depth estimation, etc.
"""  # noqa

Intro_text = """
This space showcases a demo for our paper titled "InstructCV: Towards Universal Text-to-Image Vision Generalists." We are excited to present some impressive features of our model:

1. Zero-shot Capability:
    * Our model was trained on the MS-COCO, NYUv2, Oxford-Pets, and ADE20k datasets. However, it is not limited to these datasets. You can upload any image of your choice and prompt the model to perform various vision tasks, even if they were not part of the original training set.

2. Semantic Disentangling:
    * Our model excels at handling diverse languages and instructions for different vision tasks. You can provide instructions in different languages without worrying about task confusion. The model can effectively disentangle the semantics and understand each task separately.

3. Category / Data Generalization:
    * Feel free to explore any category and experiment with images of different styles. While our model generally performs well, please note that it may not always provide optimal results for all cases. Nonetheless, we encourage you to test its capabilities across various categories and styles.

"""


example_instructions = [
                        "Please help me detect Buzz.",
                        "Please help me detect Woody's face.",
                        "Create a monocular depth map.",
]

model_id = "yulu2/InstructCV"

def main():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
    example_image = Image.open("/lustre/grp/gyqlab/lism/brt/language-vision-interface/evaluate/imgs/example.jpg").convert("RGB")
    

    def load_example():
        example_instruction = random.choice(example_instructions)
        return [example_image, example_instruction] + generate(
            example_image,
            example_instruction,
        )

    def generate(
        input_image: Image.Image,
        instruction: str,
    ):
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        if instruction == "":
            return [input_image]

        generator = torch.manual_seed(26000)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=7.5, image_guidance_scale=1.5,
            num_inference_steps=50, generator=generator,
        ).images[0]
        instruction_ = instruction.lower()
        if fnmatch(instruction_, "*segment*") or fnmatch(instruction_, "*split*") or fnmatch(instruction_, "*divide*"):
            input_image  = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR) #numpy.ndarray
            edited_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(edited_image, 127, 255, cv2.THRESH_BINARY)
            img2         = input_image.copy()
            seed         = np.random.randint(0,10000)
            np.random.seed(seed)
            colors = np.random.randint(0,255,(3))
            colors2 = np.random.randint(0,255,(3))
            contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            edited_image = cv2.drawContours(input_image,contours,-1,(int(colors[0]),int(colors[1]),int(colors[2])),3)
            for j in range(len(contours)):
                edited_image_2 = cv2.fillPoly(img2, [contours[j]], (int(colors2[0]),int(colors2[1]),int(colors2[2])))
            img_merge = cv2.addWeighted(edited_image, 0.5,edited_image_2, 0.5, 0)
            edited_image  = Image.fromarray(cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB))
        seed = 50
        text_cfg_scale = 7.5
        image_cfg_scale = 1.5
        
        return [seed, text_cfg_scale, image_cfg_scale, edited_image]


    with gr.Blocks() as demo:
#         gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">
#    InstructCV: Towards Universal Text-to-Image Vision Generalists
# </h1>""")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1.5, min_width=100):
                generate_button = gr.Button("Generate result")
            with gr.Column(scale=1.5, min_width=100):
                load_button = gr.Button("Load example")
            with gr.Column(scale=3):
                instruction = gr.Textbox(lines=1, label="Instruction", interactive=True)

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            edited_image = gr.Image(label=f"Output Image", type="pil", interactive=False)
            input_image.style(height=512, width=512)
            edited_image.style(height=512, width=512)
        
        with gr.Row(): 
            seed = gr.Number(value=26000, precision=0, label="Seed", interactive=False)
            text_cfg_scale = gr.Number(value=7.5, label=f"Text weight", interactive=False)
            image_cfg_scale = gr.Number(value=1.5, label=f"Image weight", interactive=False)


        gr.Markdown(Intro_text)
        
        load_button.click(
            fn=load_example,
            inputs=[],
            outputs=[input_image, instruction, seed, text_cfg_scale, image_cfg_scale, edited_image],
        )
        generate_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

    demo.queue(concurrency_count=1)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
