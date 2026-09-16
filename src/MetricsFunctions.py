import sklearn
from sklearn.metrics import normalized_mutual_info_score
import scipy
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, label, binary_fill_holes
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2 
import random
import sys

import skimage
from skimage.measure import regionprops# , label
from skimage.metrics import hausdorff_distance
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_erosion, binary_dilation, square
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from typing import Dict, Tuple, Any
from src.ProcessingFunctions import get_binary_masks, minmax_norm, z_norm, watershed_on_seg3D
from generative.metrics import MultiScaleSSIMMetric
import pandas as pd
#########################################################



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



#### distance between an instance and its nearest neighbor in gt and isl
def calculate_nearest_neighbor_border_distances(
    gt_labels: np.ndarray, 
    pred_labels: np.ndarray, 
    matched_df: pd.DataFrame, 
    pixel_size_um: float = 0.29
) -> pd.DataFrame:
    """
    For each matched instance in matched_df, finds its minimal border distance 
    to the nearest neighboring instance in gt_labels and pred_labels independently,
    and computes the distance error.
    """
    if matched_df is None or matched_df.empty:
        return pd.DataFrame(columns=[
            'gt_id', 'pred_id', 'gt_min_nn_dist_um', 
            'pred_min_nn_dist_um', 'nn_dist_error_um', 'abs_nn_dist_error_um'
        ])

    def extract_all_boundaries(label_img: np.ndarray) -> dict:
        """Precomputes border pixel coordinates for all objects in a label map."""
        bounds = {}
        unique_ids = np.unique(label_img)
        unique_ids = unique_ids[unique_ids != 0] # exclude background
        
        for obj_id in unique_ids:
            mask = (label_img == obj_id).astype(np.uint8)
            b_mask = find_boundaries(mask, mode='outer')
            coords = np.argwhere(b_mask) * pixel_size_um
            if len(coords) > 0:
                bounds[obj_id] = coords
        return bounds

    # Precompute all boundary coordinates in GT and Pred images
    all_gt_bounds = extract_all_boundaries(gt_labels)
    all_pred_bounds = extract_all_boundaries(pred_labels)

    results = []

    for _, row in matched_df.iterrows():
        g_id = int(row['gt_id'])
        p_id = int(row['pred_id'])

        coords_gt_target = all_gt_bounds.get(g_id)
        coords_pred_target = all_pred_bounds.get(p_id)

        if coords_gt_target is None or coords_pred_target is None:
            continue

        # --- 1. Find GT minimal border distance to ANY other GT neighbor ---
        gt_nn_dists = []
        for other_g_id, coords_gt_other in all_gt_bounds.items():
            if other_g_id == g_id:
                continue  # Skip self
            gt_nn_dists.append(np.min(cdist(coords_gt_target, coords_gt_other)))
        
        gt_min_dist = np.min(gt_nn_dists) if len(gt_nn_dists) > 0 else np.nan

        # --- 2. Find Pred minimal border distance to ANY other Pred neighbor ---
        pred_nn_dists = []
        for other_p_id, coords_pred_other in all_pred_bounds.items():
            if other_p_id == p_id:
                continue  # Skip self
            pred_nn_dists.append(np.min(cdist(coords_pred_target, coords_pred_other)))
        
        pred_min_dist = np.min(pred_nn_dists) if len(pred_nn_dists) > 0 else np.nan

        # --- 3. Compute difference ---
        if not np.isnan(gt_min_dist) and not np.isnan(pred_min_dist):
            dist_err = pred_min_dist - gt_min_dist
            results.append({
                'gt_id': g_id,
                'pred_id': p_id,
                'gt_min_nn_dist_um': gt_min_dist,
                'pred_min_nn_dist_um': pred_min_dist,
                'nn_dist_error_um': dist_err,
                'abs_nn_dist_error_um': abs(dist_err)
            })

    return pd.DataFrame(results)



def calculate_instance_hausdorff_distances(
    gt_labels: np.ndarray, 
    pred_labels: np.ndarray, 
    matched_df: pd.DataFrame, 
    pixel_size_um: float = 0.29
) -> pd.DataFrame:
    """
    Calculates the symmetric Hausdorff Distance for matched instance pairs 
    using bounding-box crops for computational efficiency.
    
    Args:
        gt_labels: 2D or 3D integer array of Ground Truth instances.
        pred_labels: 2D or 3D integer array of Predicted instances.
        matched_df: DataFrame containing matched 'gt_id' and 'pred_id' pairs.
        pixel_size_um: Physical pixel spacing factor (scales distance to microns).
        
    Returns:
        DataFrame with individual instance HD values and summary statistics.
    """
    if matched_df is None or matched_df.empty:
        return pd.DataFrame(columns=['gt_id', 'pred_id', 'hausdorff_distance_px', 'hausdorff_distance_um'])

    # Build region property lookup dicts for fast bounding box extraction
    gt_props = {p.label: p for p in regionprops(gt_labels)}
    pred_props = {p.label: p for p in regionprops(pred_labels)}

    results = []

    for _, row in matched_df.iterrows():
        g_id = int(row['gt_id'])
        p_id = int(row['pred_id'])

        if g_id not in gt_props or p_id not in pred_props:
            continue

        prop_gt = gt_props[g_id]
        prop_pred = pred_props[p_id]

        # Determine union bounding box to crop minimal sub-images
        min_row = min(prop_gt.bbox[0], prop_pred.bbox[0])
        min_col = min(prop_gt.bbox[1], prop_pred.bbox[1])
        max_row = max(prop_gt.bbox[2], prop_pred.bbox[2])
        max_col = max(prop_gt.bbox[3], prop_pred.bbox[3])

        # Slice cropped boolean masks
        gt_crop = (gt_labels[min_row:max_row, min_col:max_col] == g_id)
        pred_crop = (pred_labels[min_row:max_row, min_col:max_col] == p_id)

        # Compute symmetric Hausdorff Distance on binary crops
        hd_px = hausdorff_distance(gt_crop, pred_crop)

        if not np.isinf(hd_px):
            results.append({
                'gt_id': g_id,
                'pred_id': p_id,
                'hausdorff_distance_px': hd_px,
                'hausdorff_distance_um': hd_px * pixel_size_um
            })

    return pd.DataFrame(results)



def binary_segmentation(image: np.ndarray, min_size: int = 5) -> np.ndarray: ## (64, 64)

    ### 1. Otsu Thresholding
    thresh = threshold_otsu(image)
    binary = image > thresh # (64, 64)

    # 2. Fill internal holes (e.g., nucleoli)
    binary = binary_fill_holes(binary)
    # 3. Morphological Opening (Erosion followed by Dilation)
    # binary = binary_opening(binary, footprint=disk(1))
    
    eroded_mask = binary_erosion(binary, footprint=square(width=3))
    binary = binary_dilation(eroded_mask, footprint=square(width=3))

    
    # 4. Pre-clean small binary noise objects
    binary = remove_small_objects(binary, min_size=min_size).astype(bool) ### remove smal objects

    if not np.any(binary):
        # print('no foreground in slice')
        return np.zeros_like(image, dtype=np.int32)
    else:
        return binary.astype(int)



def instance_segmentation(binarymask: np.ndarray, peak_min_distance: int = 5, min_size: int = 5) -> np.ndarray:
    if not np.any(binarymask):
        return np.zeros_like(binarymask, dtype=np.int32)
    
    ### 2. Distance Transform 
    ### # 2D matrix where pixel values represent physical distance to the object boundary. straight-line distance to the nearest background pixel
    ### centers of nuclei end up as local intensity "peaks" because they are furthest from the edges.
    distance = distance_transform_edt(binarymask) ## 64x64
    
    ### 3. Peak Finding (Markers)
    ### Slides a search window across the distance map to locate pixels whose values are strictly greater than all neighbor pixels within the radius defined
    ### Output: An Nx2 array of coordinate pairs [[y1, x1], [y2, x2], ...], where each pair marks the center (peak) of a single detected nucleus.
    coords = peak_local_max(distance, min_distance=peak_min_distance, labels=binarymask)
    mask = np.zeros(distance.shape, dtype=bool) ###  initialize boolean matrix, (array_of_ys, array_of_xs)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True
    
    markers, _ = label(mask)
    
    ### 4. Watershed
    ###  assigns a unique, incremental integer identifier ($1, 2, 3, \dots, N$) to each contiguous region of True pixels
    ### Output: The final markers array containing unique integer IDs for each seed, with 0 denoting background.
    labels = watershed(-distance, markers, mask=binarymask)

    # labels = watershed_on_seg3D( np.expand_dims(binary.astype(int).astype('uint8'),0) ) ## my watershed
    
    labels = labels * remove_small_objects(labels > 0, min_size=min_size)

    return labels


def compute_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray): # -> Tuple[np.ndarray, list, list]:
    """
    Computes an IoU matrix between all GT and Prediction instance IDs.
    """
    gt_ids = np.unique(gt_labels)[1:]   # Exclude background (0)
    pred_ids = np.unique(pred_labels)[1:] # Exclude background (0)
    
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return np.empty((len(gt_ids), len(pred_ids))), list(gt_ids), list(pred_ids)
    
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    
    for i, g_id in enumerate(gt_ids):
        gt_mask = (gt_labels == g_id)
        for j, p_id in enumerate(pred_ids):
            pred_mask = (pred_labels == p_id)
            
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection > 0:
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou_matrix[i, j] = intersection / union
                
    return iou_matrix, list(gt_ids), list(pred_ids)


def evaluate_gt_isl( gt_img: np.ndarray, pred_img: np.ndarray, iou_threshold: float = 0.1, pixel_size_um: float = 1.0, peak_min_distance: int = 5, min_size: int = 5): # -> Dict[str, Any]:
    """
    Full pipeline: Segmentation, Matching, and Metric Calculation.
    """
    ### binary Segmentation
    gt_binarymask   = binary_segmentation(gt_img, min_size=min_size)
    pred_binarymask = binary_segmentation(pred_img, min_size=min_size)
    ### instance segmentation
    gt_seg_instances   = instance_segmentation(gt_binarymask  , peak_min_distance=5, min_size=min_size)
    pred_seg_instances = instance_segmentation(pred_binarymask, peak_min_distance=5, min_size=min_size)

    binary_iou = np.logical_and(gt_binarymask, pred_binarymask).sum() / ( np.logical_or(gt_binarymask, pred_binarymask).sum().astype('float16') + 0.00001) ## intersection / union
    
    # Extract Region Properties
    gt_props   = {prop.label: prop for prop in regionprops(gt_seg_instances)} ### area (N pixels=1), centroid com (row,col), bbox, major_axis_length & minor_axis_length, eccentricity (shape elongation) 
    pred_props = {prop.label: prop for prop in regionprops(pred_seg_instances)}

    # Step 2: Extract Matching IoU Matrix
    iou_matrix, gt_ids, pred_ids = compute_iou_matrix(gt_seg_instances, pred_seg_instances)
    
    matched_results = []
    unmatched_gt = set(gt_ids)
    unmatched_pred = set(pred_ids)
    
    # Step 3: Handle Edge Cases & Matching via Hungarian Algorithm
    if len(gt_ids) > 0 and len(pred_ids) > 0:
        # Cost matrix maximization via negative values
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        for r, c in zip(row_ind, col_ind):
            iou = iou_matrix[r, c]
            if iou >= iou_threshold:
                g_id = gt_ids[r]
                p_id = pred_ids[c]
                
                # Fetch properties
                g_p = gt_props[g_id]
                p_p = pred_props[p_id]
                
                # Area difference (Pred - GT)
                gt_area = g_p.area * (pixel_size_um ** 2)
                pred_area = p_p.area * (pixel_size_um ** 2)
                area_diff = np.abs(pred_area - gt_area)
                rel_area_diff = area_diff / gt_area
                
                # Centroid offset (y, x coordinates in skimage)
                gt_y, gt_x = g_p.centroid
                pred_y, pred_x = p_p.centroid
                
                dy = np.abs((pred_y - gt_y)) * pixel_size_um
                dx = np.abs((pred_x - gt_x)) * pixel_size_um
                centroid_dist = np.sqrt(dx**2 + dy**2)

                
                matched_results.append({
                    'gt_id': g_id,
                    'pred_id': p_id,
                    'iou': iou,
                    'gt_area_um2': gt_area,
                    'pred_area_um2': pred_area,
                    'area_diff_um2': area_diff,
                    'rel_area_diff': rel_area_diff,
                    'dx_um': dx,
                    'dy_um': dy,
                    'centroid_dist_um': centroid_dist
                })
                
                unmatched_gt.remove(g_id)
                unmatched_pred.remove(p_id)

    # Step 4: Summary Metrics
    matched_df = pd.DataFrame(matched_results)
    
    tp = len(matched_results)
    fn = len(unmatched_gt)   # Ground truth nuclei missed by prediction (iou<iou_threshold)
    fp = len(unmatched_pred) # Predicted nuclei not in ground truth
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    #### calculate min distance error between instances
    # min_dist_df = calculate_inter_instance_distances(gt_seg_instances, pred_seg_instances, matched_df, pixel_size_um = 0.29)
    min_dist_df = calculate_nearest_neighbor_border_distances(gt_seg_instances, pred_seg_instances, matched_df, pixel_size_um = 0.29)

    # print(matched_df)
    if min_dist_df.empty:
        slice_min_dist_mean = np.nan
        # plt.imshow(gt_seg_instances)
        # plt.show()
        # plt.imshow(pred_seg_instances)
        # plt.show()
        # kkk
    else:
        # slice_min_dist_mean = min_dist_df['abs_distance_error_um'].mean()  
        slice_min_dist_mean = min_dist_df['abs_nn_dist_error_um'].mean()

    ### HD
    HD_df = calculate_instance_hausdorff_distances(gt_seg_instances, pred_seg_instances, matched_df, pixel_size_um = 0.29)
    
    if HD_df.empty:
        slice_hd_mean = np.nan
    else:
        slice_hd_mean = HD_df['hausdorff_distance_um'].mean()

    

    
    return {
        'matched_pairs': matched_df,
        'summary': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_area_diff_um2': matched_df['area_diff_um2'].mean() if not matched_df.empty else np.nan,
            'mean_centroid_dist_um': matched_df['centroid_dist_um'].mean() if not matched_df.empty else np.nan,
            'slice_min_dist_mean_um': slice_min_dist_mean,
            'slice_hd_mean_um': slice_hd_mean,
        },
        'unmatched_gt_ids': list(unmatched_gt),
        'unmatched_pred_ids': list(unmatched_pred),
        'binary_iou': binary_iou
    }


def seg_watershed_iou_metrics_func(FL_images, ISL_images):
    # id = 0
    # slice = 0
    slices_area_errors = []
    id_area_errors = []
    slices_centroid_errors = []
    id_centroid_errors = []
    slices_min_dist_errors = []
    id_min_dist_errors = []
    slices_hd_errors = []
    id_hd_errors = []
    slices_f1_scores = []
    id_f1_scores = []
    slices_iou_scores = []
    id_iou_scores = []
    for id in range(FL_images.shape[0]):
        for slice in range(FL_images.shape[1]):
            slice_fl_isl_compare  = evaluate_gt_isl(FL_images[id][slice], ISL_images[id][slice], iou_threshold = 0.6, pixel_size_um = 0.29, peak_min_distance = 5, min_size = 36)
            # slice_fl_isl_compare  = evaluate_gt_isl(FL_images[id][slice], ISL_images[id][slice], iou_threshold = 0.1, pixel_size_um = 0.29, peak_min_distance = 5, min_size = 36)
                                                     
            if (~np.isinf( slice_fl_isl_compare['summary']['mean_area_diff_um2']))     and (~np.isnan( slice_fl_isl_compare['summary']['mean_area_diff_um2'])):
                slices_area_errors.append( slice_fl_isl_compare['summary']['mean_area_diff_um2'] )
            if (~np.isinf( slice_fl_isl_compare['summary']['mean_centroid_dist_um']))  and (~np.isnan( slice_fl_isl_compare['summary']['mean_centroid_dist_um'])):
                slices_centroid_errors.append( slice_fl_isl_compare['summary']['mean_centroid_dist_um'] )
            if (~np.isinf( slice_fl_isl_compare['summary']['f1_score']))               and (~np.isnan( slice_fl_isl_compare['summary']['f1_score'])):
                slices_f1_scores.append( slice_fl_isl_compare['summary']['f1_score'] )
            if (~np.isinf( slice_fl_isl_compare['summary']['slice_min_dist_mean_um'])) and (~np.isnan( slice_fl_isl_compare['summary']['slice_min_dist_mean_um'])):
                slices_min_dist_errors.append( slice_fl_isl_compare['summary']['slice_min_dist_mean_um'] )
            if (~np.isinf( slice_fl_isl_compare['summary']['slice_hd_mean_um']))       and (~np.isnan( slice_fl_isl_compare['summary']['slice_hd_mean_um'])):
                slices_hd_errors.append( slice_fl_isl_compare['summary']['slice_hd_mean_um'] )
            if (~np.isinf( slice_fl_isl_compare['binary_iou']))                        and (~np.isnan( slice_fl_isl_compare['binary_iou'])):
                slices_iou_scores.append( slice_fl_isl_compare['binary_iou'] )
        id_area_errors.append( np.array(slices_area_errors).mean() )
        id_centroid_errors.append( np.array(slices_centroid_errors).mean() )
        id_min_dist_errors.append( np.array(slices_min_dist_errors).mean() )
        id_hd_errors.append( np.array(slices_hd_errors).mean() )
        id_f1_scores.append( np.array(slices_f1_scores).mean() )
        id_iou_scores.append( np.array(slices_iou_scores).mean() )
               
    # slices_area_errors = np.array( slices_area_errors )
    # slices_centroid_errors = np.array( slices_centroid_errors )
    # slices_f1_scores = np.array( slices_f1_scores )
    # slices_min_dist_errors = np.array( slices_min_dist_errors )
    # slices_hd_errors = np.array( slices_hd_errors )
    # slices_iou_scores = np.array( slices_iou_scores )
    # return slices_iou_scores, slices_f1_scores, slices_centroid_errors, slices_min_dist_errors, slices_area_errors, slices_hd_errors 

    return id_iou_scores, id_f1_scores, id_centroid_errors, id_min_dist_errors, id_area_errors, id_hd_errors 




def calc_metrics(all_GT_images__, all_Model_images__, organelle_seg_params, rn=4):
    
    pccs, mses, msssims, jsds, mis, ious = [], [], [], [], [], []
    for i in range(len(all_GT_images__)):
        pccs.append(    pcc(all_GT_images__[i], all_Model_images__[i], rn) )
        mses.append(    mse(all_GT_images__[i], all_Model_images__[i], rn) )
        msssims.append( ms_ssim(all_GT_images__[i:i+1], all_Model_images__[i:i+1], rn) )
        jsds.append( jsd(all_GT_images__[i], all_Model_images__[i], rn) )
        mis.append( mi(all_GT_images__[i], all_Model_images__[i], rn) )
        # ious.append( seg_metrics( masks1[i:i+1], masks2[i:i+1] )[3] )    
    pccs   = np.array(pccs)
    mses   = np.array(mses)
    msssims = np.array(msssims) 
    jsds   = np.array(jsds)
    mis    = np.array(mis)
    # ious   = np.array(ious)
    
    ### aggragated on all patches (scalar)
    # pccs   = np.ones((all_GT_images__.shape[0]))  * pcc(all_GT_images__, all_Model_images__, rn)
    # mses   = np.ones((all_GT_images__.shape[0]))  * mse(all_GT_images__, all_Model_images__, rn)
    # msssims = np.ones((all_GT_images__.shape[0])) * ms_ssim(all_GT_images__, all_Model_images__, rn)
    # jsds   = np.ones((all_GT_images__.shape[0]))  * jsd(all_GT_images__ , all_Model_images__   , rn) 
    # mis    = np.ones((all_GT_images__.shape[0]))  * mi(all_GT_images__ , all_Model_images__   , rn)
    # ious   = np.ones((all_GT_images__.shape[0]))  * np.round( seg_metrics( masks1, masks2 )[3] ,rn)  

    
    ious, f1s, centroid_errors, mindist_errors, area_errors, hds  = seg_watershed_iou_metrics_func(all_GT_images__, all_Model_images__) ## shape (,n)
    return pccs, mses, msssims, jsds, mis   , ious, f1s, centroid_errors, mindist_errors, area_errors, hds                 # ious   , hds, volume_diffs, distances   # cx_diffs, cy_diffs, cz_diffs, cr_diffs




