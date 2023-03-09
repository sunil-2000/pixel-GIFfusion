import os
import warnings
import time
from pyxelate import Pyx
from skimage import io
warnings.filterwarnings("ignore")

downsample_by = 6 
upscale = 6
palette = 10
depth = 1

image_path, out_dir = "./images", './out_images'

for img_file in os.listdir(image_path):
  start_time = time.time()
  print(f'processing image: {img_file}')
  img = io.imread(f'{image_path}/{img_file}')
  pyx = Pyx(factor=downsample_by, palette=palette, upscale=upscale, depth=depth)
  pyx.fit(img)
  pixel_img = pyx.transform(img)
  io.imsave(f'{out_dir}/{img_file.split(".")[0]}.png', pixel_img)
  end_time = time.time()
  print(f'time elapsed: {end_time-start_time}')
  print('-'*50)
