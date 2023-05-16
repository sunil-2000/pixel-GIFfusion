import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from huggingface_hub import model_info
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53
        inputs_are_torch = False # NOT SURE IF THIS RIGHT

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
    
def makeGif(promt1, promt2, m_path):
    
    model_path = m_path
    info = model_info(model_path)
    model_base = info.cardData["base_model"]
    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda"
    width = 512
    height = 512
    seed = 42
    generator = torch.Generator(device=device)
    latent_noise_start = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),
	                      generator=generator.manual_seed(seed),
	                      device=device, dtype=torch.float16) 
    seed = 43
    latent_noise_end = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),
	                      generator=generator.manual_seed(seed),
	                      device=device, dtype=torch.float16)
    start_prompt = promt1
    end_prompt = promt2
    start_image = pipe(start_prompt, latents=latent_noise_start)
    end_image = pipe(end_prompt, latents=latent_noise_end)
    inputs = pipe.tokenizer([start_prompt, end_prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    prompt_embeddings = pipe.text_encoder(inputs.input_ids.to('cuda'))[0]
    num_frames = 20
    schedule = np.linspace(0, 1, num_frames)
    in_between_latents = []
    in_between_embeddings = []
    prompt_embeddings = prompt_embeddings.cpu().detach().numpy()
    for t in schedule:
        in_between_latents.append(slerp(float(t), latent_noise_start, latent_noise_end))
        in_between_embeddings.append(slerp(float(t), prompt_embeddings[0], prompt_embeddings[1]))
    
    in_between_images = []
    for i in range(len(in_between_embeddings)):
        in_between_images.append(pipe(latents=in_between_latents[i], prompt_embeds=torch.from_numpy(in_between_embeddings[i][np.newaxis, :, :])))
    images = [frame['images'][0] for frame in in_between_images]
    images[0].save(
	    f"test.gif",
	    save_all=True,
	    append_images=images[1:],
	    duration=100,
	    loop=0,
	)
	
     

	     
