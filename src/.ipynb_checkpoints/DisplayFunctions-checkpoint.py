import matplotlib.pyplot as plt
import numpy as np


def display_images(imgs, r,c, labels, show_labels='True', vmin=0, vmax=1): # 16x64x64
    fig, axs = plt.subplots(r, c, figsize=(25,2*r))
    cnt = 0
    for j in range(c):
        axs[j].imshow(imgs[cnt, :,:], cmap='gray', vmin=vmin, vmax=vmax) 
        axs[j].axis('off')
        if show_labels=='True':
            axs[j].set_title(str(labels[cnt]))
        cnt += 1
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