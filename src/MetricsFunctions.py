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
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops, label
from scipy.optimize import linear_sum_assignment
from skimage.metrics import hausdorff_distance

from src.ProcessingFunctions import get_binary_masks, minmax_norm, z_norm, watershed_on_seg3D
from generative.metrics import MultiScaleSSIMMetric

#########################################################

def get_markers_measurements(marker3D): # gets marker3D image and returns [vol1, vol2, vol3..] , [cz1, cz2, cz3...] , ....
    ### volume - number of pixels per object in the (16,64,64)
    all_obj3D_volumes = []
    all_center_z = []
    all_center_x = []
    all_center_y = []
    all_center_r = []
    
    for i in range( 1, len(np.unique(marker3D)) ): # i over all objects
        single_obj_in_img3D = np.where(marker3D==np.unique(marker3D)[np.unique(marker3D)[i]] , 1, 0)
        volume = single_obj_in_img3D.sum()
        cx, cy, cz = ndimage.center_of_mass( single_obj_in_img3D )
        cr = np.sqrt(cx**2 + cy**2 + cz**2) 
        all_obj3D_volumes.append( volume ) 
        all_center_z.append(  cz )
        all_center_x.append(  cx )
        all_center_y.append(  cy )
        all_center_r.append(  cr )
    return all_obj3D_volumes, all_center_z, all_center_x, all_center_y, all_center_r

def calculate_shortest_distances_between_objects_3d(labeled_array):
    object_labels = np.unique(labeled_array)[1:]

    if len(object_labels) < 2:
        # print("Less than two objects found. Cannot calculate object-to-object distances.")
        return 0

    # Get a list of unique object labels, excluding the background (0)
    object_labels = np.unique(labeled_array)[1:]
    
    # Dictionary to store the results: {(label1, label2): distance}
    distances = {}

    # Loop through all unique pairs of objects
    for i in range(len(object_labels)):
        label1 = object_labels[i]
        mask1 = (labeled_array == label1)
        
        # We only need to check pairs where j > i
        for j in range(i + 1, len(object_labels)):
            label2 = object_labels[j]
            mask2 = (labeled_array == label2)
            
            # --- Key Algorithm Step (Identical logic to 2D, but operating on 3D masks) ---
            
            # Calculate the Euclidean Distance Transform from the boundary of Object 1.
            # dist1[x, y, z] stores the shortest distance from voxel (x, y, z) to the 
            # nearest voxel belonging to Object 1.
            dist1 = distance_transform_edt(np.logical_not(mask1))
            
            # The shortest distance between Object 1 and Object 2 is the minimum value 
            # of the dist1 map *on* the voxels of Object 2.
            # The calculation is $\min_{v \in \text{Object } 2} (\text{dist1}[v])$
            shortest_dist = np.min(dist1[mask2])
            
            # Store the result
            distances[(label1, label2)] = shortest_dist
            
    return np.array(list(distances.values())) 

    
def avg_hd_per_slice(gt_mask, pred_mask):
    """
    Calculates the mean Hausdorff Distance across all matched instances 
    between two binary masks. Returns a single scalar.
    """
    # 1. Label connected components
    gt_labeled = label(gt_mask)
    pred_labeled = label(pred_mask)

    if isinstance(gt_labeled, tuple):
        gt_labeled = gt_labeled[0]
    if isinstance(pred_labeled, tuple):
        pred_labeled = pred_labeled[0]
        
    gt_props = regionprops(gt_labeled)
    pred_props = regionprops(pred_labeled)
    
    if len(gt_props) == 0 or len(pred_props) == 0:
        return 0.0

    # 2. Compute IoU matrix for optimal 1-to-1 matching
    iou_matrix = np.zeros((len(gt_props), len(pred_props)))
    for i, gt_reg in enumerate(gt_props):
        for j, pred_reg in enumerate(pred_props):
            # Optimization: Only compute IoU if bounding boxes overlap
            bi, bj = gt_reg.bbox, pred_reg.bbox
            if not (bi[2] < bj[0] or bi[0] > bj[2] or bi[3] < bj[1] or bi[1] > bj[3]):
                intersection = np.logical_and(gt_labeled == gt_reg.label, pred_labeled == pred_reg.label).sum()
                union = gt_reg.area + pred_reg.area - intersection
                iou_matrix[i, j] = intersection / union if union > 0 else 0

    # 3. Hungarian Matching (Optimal Assignment)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    hd_values = []
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        # Only calculate HD if there is a valid spatial match (IoU > th)
        if iou_matrix[gt_idx, pred_idx] > 0.5:
            obj_gt = (gt_labeled == gt_props[gt_idx].label)
            obj_pred = (pred_labeled == pred_props[pred_idx].label)
            
            # skimage.metrics.hausdorff_distance is symmetric by default
            dist = hausdorff_distance(obj_gt, obj_pred)
            
            if not np.isinf(dist):
                hd_values.append(dist)

    # 5. Return scalar mean
    return np.mean(hd_values) if hd_values else 0.0

def segmentation_downstream_measurements(mask3D1, mask3D2, rn=4): # (16, 64, 64)

    ### watershed 
    mask3D1_markers = watershed_on_seg3D(mask3D1)
    mask3D2_markers = watershed_on_seg3D(mask3D2)

    imgs1_binary3D_volume, imgs1_binary3D_cz, imgs1_binary3D_cx, imgs1_binary3D_cy, imgs1_binary3D_cr = get_markers_measurements( mask3D1_markers )
    imgs2_binary3D_volume, imgs2_binary3D_cz, imgs2_binary3D_cx, imgs2_binary3D_cy, imgs2_binary3D_cr = get_markers_measurements( mask3D2_markers )

    # print(imgs1_binary3D_volume, imgs1_binary3D_cr)
    # print(imgs2_binary3D_volume, imgs2_binary3D_cr)
    # kkk

    volume_diff = np.abs( np.array(imgs1_binary3D_volume).mean() ) - np.abs( np.array(imgs2_binary3D_volume).mean() ) 
    cx_diff =     np.abs( np.array(imgs1_binary3D_cx).mean()     - np.array(imgs2_binary3D_cx).mean() )  
    cy_diff =     np.abs( np.array(imgs1_binary3D_cy).mean()     - np.array(imgs2_binary3D_cy).mean() )  
    cz_diff =     np.abs( np.array(imgs1_binary3D_cz).mean()     - np.array(imgs2_binary3D_cz).mean() )  
    cr_diff =     np.abs( np.array(imgs1_binary3D_cr).mean()     - np.array(imgs2_binary3D_cr).mean() )  
    
    distances_1 = calculate_shortest_distances_between_objects_3d(mask3D1_markers)
    distances_2 = calculate_shortest_distances_between_objects_3d(mask3D2_markers)

    # print(distances_1)
    # print(distances_2)
    # print('')
        
    # if np.isnan(distances_1):
    #     distances_1 = 0
    # if np.isnan(distances_2):
    #     avg_distance_2 = 0
    avg_distance = np.abs(distances_1).mean() - np.abs(distances_2).mean()
    
    ##option b
    # results = analyze_matching_features(masks1_markers, masks2_markers, iou_threshold=0.5)
    # cr_diff =results['CoM_Distance'].values.mean()
    # volume_diff = results['Volume_Ratio'].values.mean()

    # hd = calc_avg_hd_per_id(mask3D1_markers, mask3D2_markers)

    slices1_hd = []
    for slice in range(mask3D1.shape[0]): # (16, 64, 64)
         slices1_hd.append( avg_hd_per_slice(mask3D1[slice], mask3D2[slice]) )
    hd = np.array(slices1_hd).mean()                
    
    return np.round(hd,rn), np.round(volume_diff,rn), np.round(cx_diff,rn), np.round(cy_diff,rn), np.round(cz_diff,rn), np.round(cr_diff,rn), np.round(avg_distance,rn)


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
    distances    = minmax_norm(z_norm(np.array(distances)))
    cx_diffs     = minmax_norm(z_norm(np.array(cx_diffs)))
    cy_diffs     = minmax_norm(z_norm(np.array(cy_diffs))) 
    cz_diffs     = minmax_norm(z_norm(np.array(cz_diffs))) 
    cr_diffs     = minmax_norm(z_norm(np.array(cr_diffs))) 
    

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

    
    
    return pccs, mses, msssims, jsds, mis, ious   ,   hds, volume_diffs, distances   # cx_diffs, cy_diffs, cz_diffs, cr_diffs




