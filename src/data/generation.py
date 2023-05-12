import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from datasets import load_dataset
from typing import List


class DiffusionGenerator:
    """
    use diffusion model to generate novel images for distillation
    """

    def __init__(self, model_id="kohbanye/pixel-art-style") -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()  # save GPU vram

    def load_prompts(
        self,
        n_prompts: int = 2000,
        ds_path: str = "andyyang/stable_diffusion_prompts_2m",
    ) -> List[str]:
        """
        randomly sample n_prompts from dataset
        """
        ds = load_dataset(ds_path, split="train")
        ds = ds.filter(lambda x: "by" not in x["text"])  # 800k filtered
        ds = ds.filter(lambda x: 10 <= len(x["text"]) < 140)  # 140 is median len
        k_idx = np.random.randint(low=0, high=len(ds), size=n_prompts).tolist()
        out_ds = ds.select(k_idx)
        out_ds = [prompt["text"].rsplit(",")[0] for prompt in out_ds.to_list()]
        out_ds = [prompt.rsplit(".")[0] for prompt in out_ds]
        return [f"{prompt}, pixelartstyle" for prompt in out_ds]

    def batched_text_to_image(self, prompts: List[str], batch_size=64) -> None:
        """
        generate and save images in batches
        """
        image_path = "/data/"
        meta_path = "./data/metadata.csv"
        if not os.path.exists(f".{image_path}"):
            os.makedirs(f".{image_path}")
        if not os.path.exists(meta_path):
            with open(meta_path, "a+") as f:
                f.write("file_name, prompt\n")  # headers
            f.close()

        dataloader = DataLoader(prompts, batch_size=batch_size)
        idx = 0
        for batch in tqdm(dataloader):
            images = self.pipe(batch).images
            for i, image in enumerate(images):
                image.save(f"./{image_path}{idx}.png")
                with open(meta_path, "a+") as f:
                    row = {"file_name": f"{image_path}{idx}.png", "prompt": batch[i]}
                    dict_writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    dict_writer.writerow(row)
                f.close()
                idx += 1

    def generation(self) -> None:
        """
        wrapper fn for generation
        """
        prompts = self.load_prompts()
        self.batched_text_to_image(prompts)

    def to_dataset(self, data_dir="./data", hf_path="sunilSabnis/pixelart") -> None:
        """
        convert directory to hf dataset
        """
        dataset = load_dataset("imagefolder", data_dir=data_dir)
        dataset.push_to_hub(hf_path)
