import os
import math

import matplotlib.pyplot as plt
from PIL import Image

num_images = 20
config_hash = '803521804a9a4c844495134db73291e4'

if __name__ == "__main__":

# plot the resulting image
    gt_name = 'ground_truth'
    rec_name = 'rec'
    col_num = 10
    row_num = math.ceil(num_images/col_num)
    for image_plot in range(num_images):
        gt_file = Image.open(os.path.join(f'results/{config_hash}/', f'{image_plot}_{gt_name}.png'))
        rec_file = Image.open(os.path.join(f'results/{config_hash}/', f'{image_plot}_{rec_name}.png'))
        plt.subplot(2*row_num,col_num,image_plot+1)
        plt.imshow(gt_file), plt.axis('off')
        plt.subplot(2*row_num,col_num,row_num*col_num+image_plot+1)
        plt.imshow(rec_file), plt.axis('off')
    plt.savefig('results/plot_result.pdf')