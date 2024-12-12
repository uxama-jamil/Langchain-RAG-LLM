from diffusers import StableDiffusionPipeline
import torch

class ImageGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_image(self, prompt, output_path="output.png"):
        print("Generating image for prompt:", prompt)
        image = self.pipe(prompt).images[0]
        image.save(output_path)
        return output_path
