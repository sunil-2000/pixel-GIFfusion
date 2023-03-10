import os
import warnings
import time
from pyxelate import Pyx
from skimage import io


class ImagePixelate:
    """
    Wrapper for converting images to pixel-style images
    """

    def __init__(
        self,
        factor=6,
        upscale=6,
        palette=10,
        depth=1,
        image_path="./images",
        output_path="./out_images",
        ignore_warnings=True,
    ):
        self.factor = factor
        self.upscale = upscale
        self.palette = palette
        self.depth = depth
        self.image_path = image_path
        self.output_path = output_path
        if ignore_warnings:
            warnings.filterwarnings("ignore")

    def pixelate_images(self):
        for img_file in os.listdir(self.image_path):
            start_time = time.time()
            print(f"processing image: {img_file}")
            img = io.imread(f"{self.image_path}/{img_file}")
            pyx = Pyx(
                factor=self.factor,
                palette=self.palette,
                upscale=self.upscale,
                depth=self.depth,
            )
            pyx.fit(img)
            pixel_img = pyx.transform(img)
            io.imsave(f'{self.output_path}/{img_file.split(".")[0]}.png', pixel_img)
            end_time = time.time()
            print(f"time elapsed: {end_time-start_time}")
            print("-" * 50)
