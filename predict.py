# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
import math
import cv2
import random
from fnmatch import fnmatch
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_id = "alaa-lab/InstructCV"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            cache_dir="model_cache",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        instruction: str = Input(
            description="Provide an instruction outlining the specific vision task you wish InstructCV to perform"
        ),
        text_guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
            ge=1,
            le=20,
            default=7.5,
        ),
        image_guidance_scale: float = Input(
            description="Image guidance scale is to push the generated image towards the inital image. Higher image guidance scale encourages to generate images that are closely linked to the source image, usually at the expense of lower image quality.",
            ge=1,
            default=1.5,
        ),
        num_inference_steps: int = Input(
            description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
            ge=1,
            le=500,
            default=50,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        input_image = Image.open(str(image)).convert("RGB")

        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(
            input_image, (width, height), method=Image.Resampling.LANCZOS
        )

        if instruction == "":
            return [input_image]

        generator = torch.manual_seed(seed)
        edited_image = self.pipe(
            instruction,
            image=input_image,
            guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        instruction_ = instruction.lower()

        if (
            fnmatch(instruction_, "*segment*")
            or fnmatch(instruction_, "*split*")
            or fnmatch(instruction_, "*divide*")
        ):
            input_image = cv2.cvtColor(
                np.array(input_image), cv2.COLOR_RGB2BGR
            )  # numpy.ndarray
            edited_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(edited_image, 127, 255, cv2.THRESH_BINARY)
            img2 = input_image.copy()
            seed_seg = np.random.randint(0, 10000)
            seed_seg = 11
            np.random.seed(seed_seg)
            colors = np.random.randint(0, 255, (3))
            colors2 = np.random.randint(0, 255, (3))
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            edited_image = cv2.drawContours(
                input_image,
                contours,
                -1,
                (int(colors[0]), int(colors[1]), int(colors[2])),
                3,
            )
            for j in range(len(contours)):
                edited_image_2 = cv2.fillPoly(
                    img2,
                    [contours[j]],
                    (int(colors2[0]), int(colors2[1]), int(colors2[2])),
                )
            img_merge = cv2.addWeighted(edited_image, 0.5, edited_image_2, 0.5, 0)
            edited_image = Image.fromarray(cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB))

        if fnmatch(instruction_, "*depth*"):
            edited_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGB2GRAY)
            n_min = np.min(edited_image)
            n_max = np.max(edited_image)

            edited_image = (edited_image - n_min) / (n_max - n_min + 1e-8)
            edited_image = (255 * edited_image).astype(np.uint8)
            edited_image = cv2.applyColorMap(edited_image, cv2.COLORMAP_JET)
            edited_image = Image.fromarray(
                cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)
            )

        out_path = "/tmp/out.png"
        edited_image.save(out_path)
        return Path(out_path)
