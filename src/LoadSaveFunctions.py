import tifffile
import numpy as np
import torch

from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from src.ProcessingFunctions import minmax_norm

def load_patches(main_path, organelle, imgs_name, Nimgs):
    imgs = []
    for id in range(Nimgs):
        image_ID = str(id) 
        path = main_path + 'data/' + organelle + '/' + imgs_name + '/' + image_ID + '.tiff'
        img = tifffile.imread(path)
        img = img / 255 
        imgs.append(img)
    return np.array(imgs)

def save_patches(main_path, organelle, images_to_save, images_folder):   
    img_uint8 = np.round(minmax_norm(images_to_save)*255).astype('uint8')
    for i in range(len(images_to_save)):
        image_to_save = img_uint8[i][0]
        image_to_save = image_to_save.transpose(2,0,1)
        print(image_to_save.shape, image_to_save.dtype, image_to_save.min(), image_to_save.max())
        tifffile.imwrite(main_path + 'data/' + organelle +   '/' + images_folder + '/' + str(i) + '.tiff', image_to_save)
        
        
class LoadModel:
    def __init__(self, main_path, organelle, load_model=1, timesteps=1000):
        # Default parameters
        self.device = torch.device("cuda") 
        self.model = DiffusionModelUNet(spatial_dims=3, in_channels=2, out_channels=1, num_channels=[128, 256, 256, 512], attention_levels=[False, False, False, True],                          num_res_blocks=2, num_head_channels=64,).to(self.device)
        self.tsteps = 1000
        self.scheduler = DDPMScheduler(num_train_timesteps=self.tsteps, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        
        if load_model == 1:
            self.model.load_state_dict(torch.load(main_path + "saved_models/" + organelle + "/"  + organelle + ".pth"  ))
            self.model = self.model.to(self.device)
            

        def __repr__(self):
            return (
                f"LoadModel(device={self.device}, "
                f"model={self.model}, "
                f"tsteps={self.tsteps}, "
                f"scheduler={self.scheduler}, "
                f"optimizer={self.optimizer}, ) "
               )
