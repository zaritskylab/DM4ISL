import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tifffile
import math
import cv2 

from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet # Adapted from https://github.com/huggingface/diffusers
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.networks.schedulers import DDIMScheduler
from torch.cuda.amp import GradScaler, autocast

from src.ProcessingFunctions import segmentation_pipeline, minmax_norm
from src.Params import SegmentationParams
from src.DisplayFunctions import display_images, volumetric2sequence
from src.LoadSaveFunctions import load_patches, LoadModel, save_patches

def predict_FL(DiffModel, BFinput, t_low, t_high, seed=10):
    device = torch.device("cuda") 
    DiffModel.model.eval()
    FL_preds_val      = torch.randn_like(BFinput)[0:1].to(device)  # torch.Size([1, 1, 64, 64, 16])
    all_pred_orig_val = torch.randn_like(BFinput)[0:1].to(device)
    all_pred_var_val  = torch.randn_like(BFinput)[0:1].to(device)  
    all_ids_pred_orig = torch.zeros(1,t_high-t_low,1,64,64,16).to(device) # torch.Size([1, 1, 1, 64, 64, 16])

    for id in range(BFinput.shape[0]):
        BF_id = BFinput[id:id+1]
        torch.manual_seed(seed)
        current_id_noisy = torch.randn_like(BF_id).to(device)   
        avg_pred_orig_id = torch.zeros_like(BF_id).to(device) # [1, 1, 64, 64, 16]
        all_id_pred_orig = torch.zeros_like(BF_id).to(device)
        l=0
        for ti in range(DiffModel.tsteps-1,-1,-1): # progress_bar:  # go through the noising process 999->0
            with autocast(enabled=True):
                with torch.no_grad():
                    combined_id = torch.cat((BF_id.to(device), current_id_noisy), dim=1)
                    predicted_noise_std1 = DiffModel.model(combined_id, timesteps=torch.Tensor((ti,)).to(device)) # predicts noise N(0,1)
                    current_id_noisy, pred_orig_id = DiffModel.scheduler.step(predicted_noise_std1, ti, current_id_noisy) # Predict the sample at the previous timestep ti-1 given current noisy_img at ti
                    ### average pred original 
                    if ti < t_high and  ti>= t_low :  # and ti%100==0
                        l=l+1
                        # pred_orig_id = (pred_orig_id - FL_preds_val.min()) / (FL_preds_val.max() - FL_preds_val.min())   #### minmax
                        avg_pred_orig_id = ( (l-1)*avg_pred_orig_id + pred_orig_id ) / l  ### averaging
                        # avg_pred_orig_id = (avg_pred_orig_id - FL_preds_val.min()) / (FL_preds_val.max() - FL_preds_val.min())     #### minmax
                        all_id_pred_orig = torch.cat(( all_id_pred_orig , pred_orig_id ) , dim=0 ) ## 201,1,64,64,16 per ID for calc variance
        
        FL_preds_val = torch.cat((FL_preds_val, current_id_noisy), dim=0) ## last X0's
        all_pred_orig_val = torch.cat((all_pred_orig_val, avg_pred_orig_id), dim=0) ## averaged X0's per id
        var_pred_orig_id = all_id_pred_orig[1:].var(dim=0).unsqueeze(0) ## 1,1,64,64,16
        all_pred_var_val = torch.cat((all_pred_var_val, var_pred_orig_id), dim=0) ## 1,1,64,64,16
        all_ids_pred_orig = torch.cat((all_ids_pred_orig, all_id_pred_orig[1:].unsqueeze(0)), dim=0) ################## use for calculating all timestep for 0-1000
    
    FL_preds_val = FL_preds_val[1:].cpu().detach().numpy()  
    FL_preds_val = (FL_preds_val - FL_preds_val.min()) / (FL_preds_val.max() - FL_preds_val.min())        # (8, 50, 1, 64, 64, 16)
    all_pred_orig_val = all_pred_orig_val[1:].cpu().detach().numpy()
    all_pred_orig_val = (all_pred_orig_val - all_pred_orig_val.min()) / (all_pred_orig_val.max() - all_pred_orig_val.min())          # (8, 50, 1, 64, 64, 16) 
    all_pred_var_val = all_pred_var_val[1:].cpu().detach().numpy()
    all_ids_pred_orig = all_ids_pred_orig[1:].cpu().detach().numpy()
    return FL_preds_val , all_pred_orig_val, all_pred_var_val, all_ids_pred_orig
