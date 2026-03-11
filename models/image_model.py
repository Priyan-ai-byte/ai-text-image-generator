from diffusers import StableDiffusionPipeline
import torch
import os


class ImageModel:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(device)

    def generate(self, prompt):
        image = self.pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        os.makedirs("outputs", exist_ok=True)

        path = "outputs/generated_image.png"
        image.save(path)

        return path