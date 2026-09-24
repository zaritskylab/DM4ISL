import os
import shutil

import tifffile
import numpy as np
import torch
from imutils import paths
import cv2 
import os

from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from src.ProcessingFunctions import minmax_norm
from src.Config import ORGANELLE_NAME_MAP

def load_organelle_fovs(fov_imgs_path, organelle, Nfovs=-1):
    if organelle not in ORGANELLE_NAME_MAP:
        raise ValueError(
            f"Unknown organelle '{organelle}'. "
            f"Expected one of: {list(ORGANELLE_NAME_MAP.keys())}"
        )
    fov_imgs_path = os.path.join(fov_imgs_path, ORGANELLE_NAME_MAP[organelle])
    
    imagePaths = sorted(list(paths.list_images(fov_imgs_path)))
    BFfovs = []
    FLfovs = []

    if Nfovs== -1:
        Nfovs = len(imagePaths)
    for i in range( Nfovs ): # len(imagePaths)
        fov_img = tifffile.imread(imagePaths[i]).transpose(0,2,3,1) # for full_cells_fovs
        BFfov = fov_img[0] # (624, 924, 60)
        if organelle == 'DNA':
            FLfov = fov_img[1] # (624, 924, 60)
        else:
            FLfov = fov_img[3]
        ### resize
        BFfov = cv2.resize(BFfov.astype('uint16'), (366, 244) , interpolation = cv2.INTER_NEAREST) #   cv2.INTER_AREA  cv2.INTER_LINEAR  cv2.INTER_CUBIC  cv2.INTER_LANCZOS4
        FLfov = cv2.resize(FLfov.astype('uint16'), (366, 244) , interpolation = cv2.INTER_NEAREST)
        BFfovs.append(BFfov)  
        FLfovs.append(FLfov)  
        print(imagePaths[i].split('/')[-1], BFfov.shape, '  minBF', BFfov.min() , ' maxBF', BFfov.max(), '    minFL', FLfov.min() , ' maxFL', FLfov.max() ) 
    return imagePaths, BFfovs, FLfovs


def load_patches(patches_path, organelle, image_type, Nimgs):
    imgs = []
    for id in range(Nimgs):
        image_name = f"{id:03d}_{image_type}.tiff"
        path = os.path.join(patches_path, organelle, image_type, image_name)
        img = tifffile.imread(path)
        img = img / 255
        imgs.append(img)
    return np.array(imgs)

        
def save_patches(patches_path, organelle, images_to_save, image_type):
    img_uint8 = np.round(minmax_norm(images_to_save) * 255).astype('uint8') # minmax_norm not functional
    folder_path = os.path.join(patches_path, organelle, image_type)
    os.makedirs(folder_path, exist_ok=True)
    for i in range(len(images_to_save)):
        image_to_save = img_uint8[i][0]
        image_to_save = image_to_save.transpose(2, 0, 1)
        print(image_to_save.shape, image_to_save.dtype, image_to_save.min(), image_to_save.max())
        image_name = f"{i:03d}_{image_type}.tiff"
        tifffile.imwrite(os.path.join(folder_path, image_name),image_to_save)


class LoadModel:
    def __init__(self, model_path, organelle, load_model=1, timesteps=1000):
        # Default parameters
        self.device = torch.device("cuda") 
        self.model = DiffusionModelUNet(
            spatial_dims=3, 
            in_channels=2, 
            out_channels=1, 
            num_channels=[128, 256, 256, 512], 
            attention_levels=[False, False, False, True], 
            num_res_blocks=2, 
            num_head_channels=64
        ).to(self.device)
        self.tsteps = timesteps
        self.scheduler = DDPMScheduler(num_train_timesteps=self.tsteps, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        
        if load_model == 1:
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.to(self.device)
            print('Loaded model from: ' + model_path)

            
    def __repr__(self):
        return (
            f"LoadModel(device={self.device}, "
            f"model={self.model.__class__.__name__}, "
            f"tsteps={self.tsteps}, "
            f"scheduler={self.scheduler.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__})"
        )
