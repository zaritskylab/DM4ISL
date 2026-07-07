import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tifffile
import math
import cv2 

# sys.path.append('/home/odedrot/In_silico_labelling_BF')
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet # Adapted from https://github.com/huggingface/diffusers
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.networks.schedulers import DDIMScheduler
from torch.cuda.amp import GradScaler, autocast

from src.ProcessingFunctions import segmentation_pipeline_per_patch, minmax_norm
from src.Params import SegmentationParams
from src.DisplayFunctions import display_images, volumetric2sequence
from src.LoadSaveFunctions import load_patches, LoadModel, save_patches


def predict_FL(DiffModel, BFbatch, t_low, t_high, seed=10):
    device = torch.device("cuda") 
    DiffModel.model.eval()
    xTs            = torch.randn_like(BFbatch)[0:1].cpu().detach() # original noises
    batch_x0_preds = torch.randn_like(BFbatch)[0:1].cpu().detach() # torch.Size([1, 1, 64, 64, 16])
    batch_avg_x0t  = torch.randn_like(BFbatch)[0:1].cpu().detach()
    batch_std_x0t  = torch.randn_like(BFbatch)[0:1] .cpu().detach()
    
    for id in range(BFbatch.shape[0]):
        BF_id = BFbatch[id:id+1]
        torch.manual_seed(seed)
        xt = torch.randn_like(BF_id).to(device) ### XT  
        avg_x0t = torch.zeros_like(BF_id).cpu().detach() # [1, 1, 64, 64, 16] 
        x0ts = torch.zeros_like(BF_id).cpu().detach()
        xTs = torch.cat((xTs, xt.cpu().detach()), dim=0) 
        l=0
        for ti in range(DiffModel.tsteps-1,-1,-1): # progress_bar:  # go through the noising process 999->0
            with autocast(enabled=True):
                with torch.no_grad():
                    combined_id = torch.cat((BF_id.to(device), xt), dim=1)
                    eps_pred = DiffModel.model(combined_id, timesteps=torch.Tensor((ti,)).to(device)) # predicts noise N(0,1)
                    xt, x0t = DiffModel.scheduler.step(eps_pred, ti, xt) # pred eps and x0(t)
                    ### store intermidiate x0(t) images for avg and std
                    if ti < t_high and  ti>= t_low :  # and ti%100==0
                        # l=l+1
                        # avg_x0t = ( (l-1)*avg_x0t + x0t ) / l  ### averaging
                        x0ts = torch.cat(( x0ts , x0t.cpu().detach() ) , dim=0 ) ## [N,1,64,64,16] per ID 
        
        batch_x0_preds = torch.cat((batch_x0_preds, xt.cpu().detach()), dim=0)                    ## last X0(t=0)s
        batch_avg_x0t = torch.cat((batch_avg_x0t, x0ts[1:].mean(dim=0).unsqueeze(0)), dim=0)   ## averaged X0(t)s
        batch_std_x0t = torch.cat((batch_std_x0t, x0ts[1:].std(dim=0).unsqueeze(0)   ), dim=0) ## std x0(t)s
        print(id , ' done')
    
    batch_x0_preds = batch_x0_preds[1:].numpy()  
    batch_x0_preds = (batch_x0_preds - batch_x0_preds.min()) / (batch_x0_preds.max() - batch_x0_preds.min()) # (N, 50, 1, 64, 64, 16)
    
    batch_avg_x0t = batch_avg_x0t[1:].numpy()
    batch_avg_x0t = (batch_avg_x0t - batch_avg_x0t.min()) / (batch_avg_x0t.max() - batch_avg_x0t.min())      # (N, 50, 1, 64, 64, 16) 
    
    batch_std_x0t = batch_std_x0t[1:].numpy()
    
    return batch_x0_preds, batch_avg_x0t, batch_std_x0t
