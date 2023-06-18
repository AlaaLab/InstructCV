import PIL
import requests
import torch
import cv2
import pdb
import math
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "/lustre/grp/gyqlab/lism/brt/language-vision-interface/logs/diffuser_out"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, variant='ema')
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# image = download_image(url)
img_p = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/visualization/ori_img/woody.jpg'

input_image = PIL.Image.open(img_p).convert("RGB")

width, height = input_image.size
factor = 512 / max(width, height)
factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
width = int((width * factor) // 64) * 64
height = int((height * factor) // 64) * 64
input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

prompt = "Detect Woody using a blue bounding box."
images = pipe(prompt, image=input_image, num_inference_steps=100).images
images[0].save("5555.jpg")

