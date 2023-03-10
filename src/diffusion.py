import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


class Diffusion:
    "Class for applying a stable diffusion model"

    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        """
        initialize model --> later change for more customizability
        must be connected to a GPU runtime
        """
        self.model_id = model_id
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()  # save GPU vram

    def text_to_image(self, prompts, img_name="diffusion_img", output_dir="./"):
        """
        prompt[str]: list of prompts (treat as batch)
        """
        images = self.pipe(prompts).images
        for i, image in enumerate(images):
            image.save(f"{output_dir}/{i}_{img_name}.png")
