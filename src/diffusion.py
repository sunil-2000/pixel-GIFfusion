import torch
import numpy as np
from PIL import Image
from PIL.Image import Image
from huggingface_hub import model_info
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from typing import List
from torch import Tensor


class Diffusion:
    "Class for applying a stable diffusion model"

    def __init__(
        self,
        model_id: str = "jainr3/sd-diffusiondb-pixelart-v2-model-lora",
        custom: bool = True,
    ):
        """
        model_id: hf model path
        custom: whether using custom fine-tuned model with Llora
        """
        if custom:
            info = model_info(model_id)
            model_base = info.cardData["base_model"]
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_base, torch_dtype=torch.float16
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.pipe.unet.load_attn_procs(model_id)
        else:
            self.model_id = model_id
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        self.pipe = self.pipe.to("cuda")

    def text_to_image(
        self, prompts: List[str], img_name="diffusion_img", output_dir="./"
    ) -> None:
        """
        prompt[str]: list of prompts (treat as batch)
        """
        images = self.pipe(prompts).images
        for i, image in enumerate(images):
            image.save(f"{output_dir}/{i}_{img_name}.png")

    def text_to_gif(
        self, prompts: List[str], img_name="diffusion_img", output_dir="./"
    ) -> List[Tensor]:
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

    def _latent_lst_to_gif(
        self, latent_states: Tensor, img_name: str, output_dir: str
    ) -> None:
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
        prompt1 = input("First prompt: ")
        prompt2 = input("Second prompt: ")
        images = self.gif_step(prompt1, prompt2)
        images[0].save(
            f"morphed.gif",
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
            )

    def gif_chain(self, prompts: List[str], speed: int = 200, num_frames:int=40) -> None:
        """
        chain prompts in sequence to create larger gif
        """
        images = []
        for i in tqdm(range(len(prompts) - 1)):  # process in pairs
            images.extend(self.gif_step(prompts[i], prompts[i + 1], num_frames=num_frames))

        images[0].save(
            f"morphed.gif",
            save_all=True,
            append_images=images[1:],
            duration=speed,
            loop=0,
        )

    def gif_step(
        self, start_prompt: str, end_prompt: str, num_frames: int = 40
    ) -> List[Image]:
        # Hard coded image dims
        device = "cuda"
        width, height = 768, 768

        # Generate the latents for the start and end prompt(random noise from gaussian distribution)
        seed = 42
        generator = torch.Generator(device=device)
        latent_noise_start = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator.manual_seed(seed),
            device=device,
            dtype=torch.float16,
        )

        seed = 43
        latent_noise_end = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator.manual_seed(seed),
            device=device,
            dtype=torch.float16,
        )

        # Convert prompts to CLIP Embeddings for interpolation
        inputs = self.pipe.tokenizer(
            [start_prompt, end_prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeddings = self.pipe.text_encoder(inputs.input_ids.to("cuda"))[0]

        # Interpolate the start and end latent noise tensors and prompt embeddings
        schedule = np.linspace(0, 1, num_frames)
        in_between_latents, in_between_embeddings = [], []
        prompt_embeddings = prompt_embeddings.cpu().detach().numpy()

        for t in schedule:
            in_between_latents.append(
                self._slerp(float(t), latent_noise_start, latent_noise_end)
            )
            in_between_embeddings.append(
                self._slerp(float(t), prompt_embeddings[0], prompt_embeddings[1])
            )

        # Generate the interpolated images using the embeddings which will be used as the gif's transition images
        in_between_images = []
        for i in range(len(in_between_embeddings)):
            in_between_images.append(
                self.pipe(
                    latents=in_between_latents[i],
                    prompt_embeds=torch.from_numpy(
                        in_between_embeddings[i][np.newaxis, :, :]
                    ),
                )
            )

        images = [frame["images"][0] for frame in in_between_images]
        return images

    def _slerp(
        self, t: float, v0: np.array, v1: np.array, DOT_THRESHOLD: float = 0.9995
    ):
        """helper function to spherically interpolate two arrays v1 v2"""
        # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53
        inputs_are_torch = False  # NOT SURE IF THIS RIGHT

        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2
