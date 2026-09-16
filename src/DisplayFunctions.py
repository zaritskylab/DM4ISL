import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import math


def display_images(imgs, r,c, labels, show_labels='True', rect='none', vmin=0, vmax=1): # 16x64x64
    fig, axs = plt.subplots(r, c, figsize=(25,2*r))
    cnt = 0
    for j in range(c):
        axs[j].imshow(imgs[cnt, :,:], cmap='gray', vmin=vmin, vmax=vmax) 
        # axs[0].add_patch(  Rectangle((0, 0), imgs[cnt, :,:].shape[1], imgs[cnt, :,:].shape[0], linewidth=10, edgecolor='blue', facecolor='none')  )
        axs[j].axis('off')
        if show_labels=='True':
            axs[j].set_title(str(labels[cnt]))
        cnt += 1
    if rect=='red':
        fig.patches.append( Rectangle( (0.11, 0.15), 0.8, 0.7, linewidth=4, edgecolor='red', facecolor='none', transform=fig.transFigure, clip_on=False ) )
    if rect=='green':
        fig.patches.append( Rectangle( (0.11, 0.15), 0.8, 0.7, linewidth=4, edgecolor='green', facecolor='none', transform=fig.transFigure, clip_on=False ) )
    plt.show()


def display_images_grid(images, values, max_cols=5, figsize_per_col=3, figsize_per_row=3, vmin=-1, vmax=1):
    """
    Displays N images in a grid layout in Jupyter Lab.
   
    Parameters:
    - images: List or array of N images (2D grayscale or 3D RGB arrays).
    - values: List or array of N values/labels to display above each image.
    - max_cols: Maximum number of images to show per row.
    - figsize_per_col: Width in inches per subplot column.
    - figsize_per_row: Height in inches per subplot row.
    """
    N = len(images)
    if N == 0:
        print("No images to display.")
        return
       
    if N != len(values):
        raise ValueError(f"Length mismatch: {N} images provided, but {len(values)} values provided.")
   
    # Determine grid structure
    cols = min(N, max_cols)
    rows = math.ceil(N / cols)
   
    # Initialize figure and axes grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_col, rows * figsize_per_row)    )
   
    # Flatten axes array for consistent indexing regardless of grid size
    axes = np.array(axes).reshape(-1)
   
    # Plot each image and its title
    for i in range(N):
        ax = axes[i]
       
        # Display grayscale images using 'gray' colormap
        if images[i].ndim == 2:
            ax.imshow(images[i], cmap='gray' , vmin=vmin, vmax=vmax)
        else:
            ax.imshow(images[i], vmin=vmin, vmax=vmax)
           
        ax.set_title(str(values[i]), fontsize=11, fontweight='bold')
        ax.axis('off')  # Hide pixel coordinate axes
       
    # Turn off axes for any remaining empty grid slots
    for j in range(N, len(axes)):
        axes[j].axis('off')
       
    plt.tight_layout()
    plt.show()
    
    
    
def volumetric2sequence(imgs3D): # 16,64,64
    # if imgs3D.shape[-1] == 1:
    #     seq_image = imgs3D[0,:,:]
    # else:
    seq_image = np.zeros((imgs3D.shape[1],imgs3D.shape[2]))
    for i in range(imgs3D.shape[0]):
        seq_image = np.concatenate((seq_image, imgs3D[i,:,:]) , axis = -1)  # 16,1,64,64,1
    seq_image = seq_image[:,64:]
    return seq_image