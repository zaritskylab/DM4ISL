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

def load_organelle_fovs(fov_imgs_path, organelle, Nfovs=-1):
    # fov_imgs_path = main_path + '/'+ fovs_path
    if organelle == 'Nuclear envelope':
         fov_imgs_path = fov_imgs_path + '/'+ 'Nuclear-envelope/'
    if organelle == 'Actin filament':
        fov_imgs_path = fov_imgs_path + '/'+ 'Actin-filaments/'
    if organelle == 'Microtubules':
        fov_imgs_path = fov_imgs_path + '/'+ 'Microtubules/'
    if organelle == 'Mitochondria':
        fov_imgs_path = fov_imgs_path + '/'+ 'Mitochondria/'
    if organelle == 'Nucleoli':
        fov_imgs_path = fov_imgs_path + '/'+ 'Nucleolus-(Dense-Fibrillar-Component)/'
    if organelle == 'DNA':
        fov_imgs_path = fov_imgs_path + '/'+ 'Mitochondria/'
    
    imagePaths = sorted(list(paths.list_images(fov_imgs_path)))
    BFfovs = []
    FLfovs = []
    i = 0  
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


def load_patches(patches_path, organelle, imgs_name, Nimgs):
    imgs = []
    for id in range(Nimgs):
        image_ID = str(id) 
        path = patches_path + '/'+ organelle + '/' + imgs_name + '/' + image_ID + '.tiff'
        img = tifffile.imread(path)
        img = img / 255 
        imgs.append(img)
    return np.array(imgs)

def save_patches(patches_path, organelle, images_to_save, images_folder):   
    img_uint8 = np.round(minmax_norm(images_to_save)*255).astype('uint8')
    for i in range(len(images_to_save)):
        image_to_save = img_uint8[i][0]
        image_to_save = image_to_save.transpose(2, 0, 1)
        print(image_to_save.shape, image_to_save.dtype, image_to_save.min(), image_to_save.max())
        
        folder_path = patches_path + '/' + organelle +   '/' + images_folder + '/'
        os.makedirs(folder_path, exist_ok=True)
        tifffile.imwrite(os.path.join(folder_path, str(i) + '.tiff'), image_to_save)
        
        
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
            
            
    # FIXED: Moved this back out so it is a proper class method
    def __repr__(self):
        return (
            f"LoadModel(device={self.device}, "
            f"model={self.model.__class__.__name__}, "
            f"tsteps={self.tsteps}, "
            f"scheduler={self.scheduler.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__})"
        )
