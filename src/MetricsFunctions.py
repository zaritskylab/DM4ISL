import skimage
import sklearn
from sklearn.metrics import normalized_mutual_info_score
import scipy
from scipy import ndimage
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
import torch
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2 
import random
import sys

from src.ProcessingFunctions import segmentation_downstream_measurements, get_binary_masks, minmax_norm, z_norm
from generative.metrics import MultiScaleSSIMMetric

#########################################################

def seg_metrics(y_true, y_pred):
    wtp = 1
    wfn = 1
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(np.abs(y_pred * y_true)) # TP = intersection of GT and pred
    mask_sum =     np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred)) # GT_area + pred_area
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot
    TP = intersection
    FN = np.sum(np.abs(y_true)) - TP # pixels=1 in GT but 0 in pred
    FP = np.sum(np.abs(y_pred)) - TP # pixels=1 in pred but 0 in GT
    smooth = 0.00001
    # iou = (intersection + smooth) / (union + smooth)
    iou = (wtp*TP + smooth) / (wtp*TP + FP + wfn*FN + smooth)
    # dice = 2*(intersection + smooth)/(mask_sum + smooth)
    dice = 2*(wtp*TP + smooth)/(2*wtp*TP + wfn*FN +FP + smooth)
    Precision = wtp*TP / (wtp*TP + FP + smooth)
    Recall = wtp*TP / (wtp*TP + wfn*FN + smooth) # TPR
    F1score = 2*wtp*TP / (2*wtp*TP + FP + wfn*FN + smooth)
    return TP.mean(), FN.mean(), FP.mean(),  iou.mean() , dice.mean(), Precision.mean(), Recall.mean(), F1score.mean()


def mi(imgs1, imgs2, rn=3, bins=20):
    eps = 0.000000000001
    hgram, x_edges, y_edges = np.histogram2d( imgs1.ravel(),imgs2.ravel(), bins=bins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    Hx = -np.sum( px*np.log2(px+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=1) * np.log2(joint_prob_mat.sum(axis=1)+eps) )
    Hy = -np.sum( py*np.log2(py+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=0) * np.log2(joint_prob_mat.sum(axis=0)+eps) )
    Hxy = -np.sum(pxy * np.log2(pxy+eps)) 
    mi = Hx + Hy - Hxy
    return np.round(mi, rn) 

def ms_ssim(imgs1, imgs2, rn=3):
    return np.round( MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)( torch.tensor(imgs1), torch.tensor(imgs2) ).mean().numpy(), rn)

def pcc(imgs1, imgs2, rn=3):
    return np.round( stats.pearsonr( imgs1.flatten() , imgs2.flatten() )[0], rn)

def mse(imgs1, imgs2, rn=3):
    return np.round( np.mean(np.abs( imgs1.flatten() - imgs2.flatten() )**2), rn)

def jsd(imgs1, imgs2, rn=3):
    return np.round( jensenshannon( imgs1.flatten(), imgs2.flatten(), base=2 ), rn)



def calc_metrics(all_GT_images__, all_Model_images__, organelle_seg_params, rn=4):
    masks1, masks2 = get_binary_masks(all_GT_images__ , all_Model_images__, organelle_seg_params, rn)  ## (128, 16, 64, 64), (128, 16, 64, 64)
    
    ### metric per patch and then avarage  
    hds, volume_diffs, cx_diffs, cy_diffs, cz_diffs, cr_diffs, distances = [], [], [], [], [], [], []
    for i in range(len(all_GT_images__)):
        hd, volume_diff, cx_diff, cy_diff, cz_diff, cr_diff, distance = segmentation_downstream_measurements(masks1[i], masks2[i])
        hds.append( hd ), volume_diffs.append( volume_diff ), cx_diffs.append( cx_diff ), cy_diffs.append( cy_diff ), cz_diffs.append( cz_diff ), cr_diffs.append( cr_diff ), distances.append( distance )    
    hds          = minmax_norm(z_norm(np.array(hds)))
    volume_diffs = minmax_norm(z_norm(np.array(volume_diffs)))
    cx_diffs     = minmax_norm(z_norm(np.array(cx_diffs)))
    cy_diffs     = minmax_norm(z_norm(np.array(cy_diffs))) 
    cz_diffs     = minmax_norm(z_norm(np.array(cz_diffs))) 
    cr_diffs     = minmax_norm(z_norm(np.array(cr_diffs))) 
    distances    = minmax_norm(z_norm(np.array(distances)))
    

    # pccs, mses, msssims, jsds, mis, ious = [], [], [], [], [], []
    # for i in range(len(all_GT_images__)):
    #     pccs.append(    pcc(all_GT_images__[i], all_Model_images__[i], rn) )
    #     mses.append(    mse(all_GT_images__[i], all_Model_images__[i], rn) )
    #     msssims.append( ms_ssim(all_GT_images__[i:i+1], all_Model_images__[i:i+1], rn) )
    #     jsds.append( jsd(all_GT_images__[i], all_Model_images__[i], rn) )
    #     mis.append( mi(all_GT_images__[i], all_Model_images__[i], rn) )
    #     ious.append( seg_metrics( masks1[i:i+1], masks2[i:i+1] )[3] )    
    # pccs   = np.array(pccs)
    # mses   = np.array(mses)
    # msssims = np.array(msssims) 
    # jsds   = np.array(jsds)
    # mis    = np.array(mis)
    # ious   = np.array(ious)
    

    ### aggragated on all patches (scalar)
    pccs   = np.ones((all_GT_images__.shape[0]))  * pcc(all_GT_images__, all_Model_images__, rn)
    mses   = np.ones((all_GT_images__.shape[0]))  * mse(all_GT_images__, all_Model_images__, rn)
    msssims = np.ones((all_GT_images__.shape[0])) * ms_ssim(all_GT_images__, all_Model_images__, rn)
    jsds   = np.ones((all_GT_images__.shape[0]))  * jsd(all_GT_images__ , all_Model_images__   , rn) 
    mis    = np.ones((all_GT_images__.shape[0]))  * mi(all_GT_images__ , all_Model_images__   , rn)
    ious   = np.ones((all_GT_images__.shape[0]))  * np.round( seg_metrics( masks1, masks2 )[3] ,rn)   

    
    
    return pccs, mses, msssims, jsds, mis, ious   ,   hds, volume_diffs, cx_diffs, cy_diffs, cz_diffs, cr_diffs, distances  




