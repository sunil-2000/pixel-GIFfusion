import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from MorphImages import make_gif


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

    def text_to_gif(self, prompts, img_name="diffusion_img", output_dir="./"):
        """
        return image collection (GIF) of latent states from generative process
        """
        # single image as of now
        latent_states = []

        def callback_fn(i, t, latent):
            latent_states.append(latent)

        self.pipe(prompts, callback=callback_fn)
        self._latent_lst_to_gif(latent_states, img_name, output_dir=output_dir)
        return latent_states

    def _latent_lst_to_gif(self, latent_states, img_name, output_dir):
        """
        convert latent state list to GIF
        """
        img_latent_states = [
            np.array(latent_state * 255, dtype="uint8").squeeze()
            for latent_state in latent_states
        ]
        images = [Image.fromarray(frame) for frame in img_latent_states]
        images[0].save(
            f"{output_dir}{img_name}.gif",
            save_all=True,
            append_images=images[1:],
            duration=50,
            loop=0,
        )

    def prompt_to_gif(self):
        """
        asks user for two prompts, creates and saves gif that morphs between 
        the two images in current directory
        """
        prompt1 = input('First prompt: ')
        prompt2 = input('Second prompt: ')
        make_gif(prompt1, prompt2)