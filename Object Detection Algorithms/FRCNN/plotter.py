import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random
from tqdm import tqdm
def plotter(image: np.array, bbox):
    m_c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(figsize=(10,10))
    plt.imshow(image[0, :, :, :].numpy().transpose(), origin='lower')  # Set origin to 'lower'
    for box in tqdm(bbox):
        x, y, w, h = box
        patch1 = Rectangle((x, y), w, h, edgecolor=random.choice(m_c), facecolor='None')
        ax = plt.gca()
        ax.add_patch(patch1)

    # Adjust y-axis limits if needed
    plt.ylim([0, image.shape[2]])  # Assuming image.shape[2] is the height of the image
    plt.show()