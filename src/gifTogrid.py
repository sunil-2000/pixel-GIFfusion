import imageio
import matplotlib.pyplot as plt


gif_path = '../images/test-3.gif'
gif_frames = imageio.mimread(gif_path)

fig, axs = plt.subplots(4, 10, figsize=(10, 4))

for i, ax in enumerate(axs.flat):
    ax.imshow(gif_frames[i], cmap='gray')
    ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig('grid.png')
