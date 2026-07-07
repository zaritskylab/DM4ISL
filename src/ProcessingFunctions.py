import skimage
from skimage.morphology import remove_small_objects     # function for post-processing (size filter)
from skimage.metrics import hausdorff_distance
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import regionprops, label
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
import cv2 
import random
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import r2_score, explained_variance_score
import scipy
from scipy.ndimage import zoom
from scipy.ndimage import distance_transform_edt, label
from scipy import ndimage
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom, generate_binary_structure
from scipy.stats import spearmanr, skew, kurtosis 
from glrlm import GLRLM
from scipy.stats import mannwhitneyu
from scipy.optimize import linear_sum_assignment
import torch
from src.Params import organelle_parameters


def train_test_split(fovs_names, BFfovs, FLfovs, test_perc = 0.1, seed=35):
    np.random.seed( seed ) 
    test_ids = np.random.randint(len(BFfovs), size=int(np.round(len(BFfovs)*test_perc)))
    train_ids = np.array(list(set( list(np.arange(len(BFfovs))) ) - set(test_ids))).astype(int)
    train_fovs_names, test_fovs_names, train_BFfovs, train_FLfovs, test_BFfovs, test_FLfovs = [], [], [], [], [], []
    for i in range(len(BFfovs)):
        if i in test_ids:
            test_BFfovs.append(BFfovs[i])  
            test_FLfovs.append(FLfovs[i])
            test_fovs_names.append(fovs_names[i])
        else:
            train_BFfovs.append(BFfovs[i])  
            train_FLfovs.append(FLfovs[i])
            train_fovs_names.append(fovs_names[i])
    
    print('N train' , len(train_BFfovs))
    print('N test'  , len(test_BFfovs))
    return train_fovs_names, train_BFfovs, train_FLfovs, test_fovs_names, test_BFfovs, test_FLfovs

def create_random_3Dpatches(signals_data, targets_data, Npatches, patch_sizeX, patch_sizeY, patch_sizeZ, train=True, seed=5):
    if train:
        random_ids = np.random.randint(len(signals_data), size=Npatches)
    else:
        np.random.seed(seed)
        random_ids = np.random.randint(len(signals_data), size=Npatches)
    
    BFbatch = []
    FLbatch = []
    for i in range(len(random_ids)):
        BF3D = signals_data[random_ids[i]].squeeze()  # (624,924,75) choose one of the 3D samples
        FL3D = targets_data[random_ids[i]].squeeze() # (624,924,75) choose one of the 3D samples
        ## choose random xyz inside  the zone which is possible to crop a patch out of
        y = np.random.randint(BF3D.shape[0] - patch_sizeY, size=1)[0]        
        x = np.random.randint(BF3D.shape[1] - patch_sizeX, size=1)[0]
        z = np.random.randint(BF3D.shape[2] - patch_sizeZ, size=1)[0] 
       
        BFpatch3D = BF3D[y:y + patch_sizeY, x:x + patch_sizeX, z:z + patch_sizeZ]
        FLpatch3D = FL3D[y:y + patch_sizeY, x:x + patch_sizeX, z:z + patch_sizeZ]
         
        ## unet needs float32 not float64
        BFpatch3D = BFpatch3D.astype('float32')
        FLpatch3D = FLpatch3D.astype('float32')  
        
        BFbatch.append(BFpatch3D)
        FLbatch.append(FLpatch3D)
            
    BFbatch = torch.tensor(np.array(BFbatch)).unsqueeze(1)
    FLbatch = torch.tensor(np.array(FLbatch)).unsqueeze(1)
    return BFbatch, FLbatch

def preprocess_patches(BFbatch, FLbatch, organelle):
    FLstd_TH, BFmaxGL_clip, BFminGL_clip, FLmaxGL_clip, FLminGL_clip = organelle_parameters(organelle)
    BFbatch = torch.clip(BFbatch, BFminGL_clip, BFmaxGL_clip)
    FLbatch = torch.clip(FLbatch, FLminGL_clip, FLmaxGL_clip)
    FLbatch_stds = torch.min( torch.std(FLbatch, axis=(1,2,3)) , axis=1)[0] # min std of slices within each 3D patch 
    FLbatch = FLbatch[torch.where(FLbatch_stds > FLstd_TH)[0]]
    BFbatch = BFbatch[torch.where(FLbatch_stds > FLstd_TH)[0]]
    # ### normalize batch
    BFbatch = (BFbatch - BFminGL_clip) / (BFmaxGL_clip - BFminGL_clip)
    FLbatch = (FLbatch - FLminGL_clip) / (FLmaxGL_clip - FLminGL_clip)
    return BFbatch, FLbatch

def run_augmentations(imagesA, imagesB):
    ### Hor LR flip
    if random.randint(0, 10) > 3:
        imagesA = torch.flip(imagesA, [3])
        imagesB = torch.flip(imagesB, [3])
    # ### Ver UD flip
    if random.randint(0, 10) > 3:
        imagesA = torch.flip(imagesA, [2])
        imagesB = torch.flip(imagesB, [2])
    ### rotate
    if random.randint(0, 10) > 3:
        Nrot = random.randint(1, 3)
        imagesA = torch.rot90(imagesA, Nrot, [2,3])
        imagesB = torch.rot90(imagesB, Nrot, [2,3])
            
    return imagesA, imagesB
    
def volumetric2sequence(imgs3D): # 16,64,64
    # if imgs3D.shape[-1] == 1:
    #     seq_image = imgs3D[0,:,:]
    # else:
    seq_image = np.zeros((imgs3D.shape[1],imgs3D.shape[2]))
    for i in range(imgs3D.shape[0]):
        seq_image = np.concatenate((seq_image, imgs3D[i,:,:]) , axis = -1)  # 16,1,64,64,1
    seq_image = seq_image[:,64:]
    return seq_image

def z_norm(data):
    return data # (data - data.mean())  / (data.std() + 0.0001)

def minmax_norm(data):
    return data # (data - data.min())  / (data.max() - data.min())

def segment3D(img3D, organelle_th=65): # should be 0-255 uint8
    # img3D = (img3D*255).astype('uint8') # 16,64,64
    ### thresholding 
    if organelle_th == -2: # one th for all slices
        seq = volumetric2sequence( img3D ) # 
        organelle_th , _ = cv2.threshold(seq.astype('uint8'),  0, 255, cv2.THRESH_OTSU)
    img3D_binary = []
    for i in range(img3D.shape[0]):
        if organelle_th == -1:
            organelle_th_, bimg = cv2.threshold(img3D[i,:,:],  0, 255, cv2.THRESH_OTSU)
        else:
            organelle_th_, bimg = cv2.threshold( img3D[i,:,:] ,organelle_th ,255,cv2.THRESH_BINARY ) # thresh.shape=64,64, ret=35 cv2.THRESH_BINARY+ cv2.THRESH_OTSU
        img3D_binary.append(bimg) 
    img3D_binary = np.array(img3D_binary).astype('uint8') # 16,64,64
    return organelle_th, img3D_binary

def erode_dilate(binary3D, k=2, iterations=1):
    seg_morph_slices = []
    for i in range(binary3D.shape[0]):
        binary2D = binary3D[i,:,:].astype('uint8')
        seg_morph = binary2D.astype(np.uint8)  
        seg_morph = cv2.erode( seg_morph,  np.ones((2, 2), np.uint8)  , iterations=1)
        seg_morph = cv2.dilate(seg_morph,  np.ones((2, 2), np.uint8)  , iterations=1)
        seg_morph_slices.append(seg_morph.astype('uint8'))
    seg_morph_slices = np.array(seg_morph_slices) # 16,64,64
    return seg_morph_slices


def run_filter(img3D, filter_kernel = 5, sigma = 50, filter_type='median'): # should be 0-255 uint8
    img3D = img3D.astype('uint8') # 16,64,64
    ### denoising filter 
    img3D_denoised = []
    for i in range(img3D.shape[0]):
        if filter_type == 'median':
            # img3D_denoised.append( cv2.medianBlur( img3D[i,:,:]   , filter_kernel ) ) # 3-for nucleoli, NUcEnv
            img3D_denoised.append( ndimage.median_filter(img3D[i,:,:], size=filter_kernel) )
        elif filter_type == 'bilateral':
            img3D_denoised.append( cv2.bilateralFilter(img3D[i,:,:] , filter_kernel, sigma, sigma) ) 
    img3D_denoised = np.array(img3D_denoised) # 16,64,64
    return img3D_denoised.astype('uint8')

def fill_holes(binary3D, k=3):
    seg_morph_slices = []
    for i in range(binary3D.shape[0]):
        binary2D = binary3D[i,:,:].astype('uint8')
        seg_morph = scipy.ndimage.binary_fill_holes(binary2D , np.ones((k,k))).astype('uint8') # (doesnt fill holes touching boundaries) 
        seg_morph_slices.append(seg_morph.astype('uint8'))
    seg_morph_slices = np.array(seg_morph_slices)*255 # 16,64,64 0-1 --> 0-255
    return seg_morph_slices

def fill_holes_boarders(binary3D, k=3):
    seg_morph_slices = []
    for i in range(binary3D.shape[0]):
        binary2D = binary3D[i,:,:].astype('uint8')
        
        seg_morph = scipy.ndimage.binary_fill_holes(binary2D , np.ones((k,k))).astype('uint8') # (doesnt fill holes touching boundaries) 
        ## pad edges to fill holes on the borders
        tmp = np.ones((seg_morph.shape[0]+2, seg_morph.shape[1]+1))
        tmp[1:-1 , :-1] = seg_morph
        seg_morph_ = scipy.ndimage.binary_fill_holes(tmp , np.ones((k,k))).astype('uint8')
        seg_morph_ = seg_morph_[1:-1 , :-1]
        seg_morph_ = skimage.morphology.remove_small_objects(seg_morph_, min_size = 50) # filter small particle
        tmp = np.ones((seg_morph_.shape[0]+2, seg_morph_.shape[1]+1))
        tmp[1:-1 , 1:] = seg_morph_
        seg_morph_ = scipy.ndimage.binary_fill_holes(tmp , np.ones((k,k))).astype('uint8')
        seg_morph = seg_morph_[1:-1 , 1:]
        seg_morph = skimage.morphology.remove_small_objects(seg_morph, min_size = 50) # filter small particles
        seg_morph = (seg_morph*255).astype('uint8')
        
        seg_morph_slices.append(seg_morph.astype('uint8'))
    seg_morph_slices = np.array(seg_morph_slices) # 16,64,64
    return seg_morph_slices

def remove_small_objects_func(binary3D, k=3):
    seg_morph_slices = []
    for i in range(binary3D.shape[0]):
        binary2D = binary3D[i,:,:].astype('uint8')
        # seg_morph = skimage.morphology.remove_small_objects(seg_morph, min_size = 100) # filter small particles 
        contours, hierarchy = cv2.findContours( binary2D.astype('uint8') , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 3rd input: CHAIN_APPROX_SIMPLE or CHAIN_APPROX_NONE
        mask = np.zeros((binary3D.shape[1],binary3D.shape[2]))
        for j in range(len(contours)):
            if cv2.contourArea(contours[j]) > 15:  
                mask = cv2.fillPoly(mask, [contours[j]] , 255)          
        seg_morph =  mask   
        seg_morph_slices.append(seg_morph.astype('uint8'))
    seg_morph_slices = np.array(seg_morph_slices) # 16,64,64
    return seg_morph_slices

## segments a single 3D patch 
def segmentation_pipeline_per_patch(image3D, filter_type, k1, k2, k3, filter_kernel, sigma, organelle_th, do_erode_dilate, do_remove_small_objects, do_fill_holes, do_fill_holes_boarders):
    image_filter = run_filter((image3D*255), filter_kernel=filter_kernel, sigma=sigma, filter_type=filter_type) # 0-255
    seg_th, seg_stack = segment3D(image_filter, organelle_th=organelle_th)     # 0-255  
    if do_erode_dilate == True:
        seg_stack = erode_dilate(seg_stack, k=k1, iterations=1)                # 0-255
    if do_remove_small_objects == True:
        seg_stack = remove_small_objects_func(seg_stack, k=k2)                 # 0-255
    if do_fill_holes == True:
        seg_stack = fill_holes(seg_stack, k=k1)                                # 0-255
    if do_fill_holes_boarders == True:
        seg_stack = fill_holes_boarders(seg_stack, k=k3)                       # 0-255
    return seg_th, seg_stack



def get_binary_masks(imgs1, imgs2, organelle_seg_params, rn=3):   ## segmentation per patch3D
    ## general segmentation metrics
    imgs1_seg_morph_slices, imgs2_seg_morph_slices = np.zeros_like(imgs1), np.zeros_like(imgs2)
    for i in range(imgs1.shape[0]):
        imgs1_th, imgs1_seg_morph_slices_ = segmentation_pipeline_per_patch(imgs1[i] , filter_type=organelle_seg_params.filter_type, k1=organelle_seg_params.k1, k2=organelle_seg_params.k2,    
        k3=organelle_seg_params.k3, filter_kernel=organelle_seg_params.filter_kernel, sigma=organelle_seg_params.sigma, organelle_th=organelle_seg_params.organelle_th, 
        do_erode_dilate=organelle_seg_params.do_erode_dilate, do_remove_small_objects=organelle_seg_params.do_remove_small_objects, do_fill_holes=organelle_seg_params.do_fill_holes, 
        do_fill_holes_boarders=organelle_seg_params.do_fill_holes_boarders)  
        
        imgs2_th, imgs2_seg_morph_slices_ = segmentation_pipeline_per_patch(imgs2[i] , filter_type=organelle_seg_params.filter_type, k1=organelle_seg_params.k1, k2=organelle_seg_params.k2, 
        k3=organelle_seg_params.k3, filter_kernel=organelle_seg_params.filter_kernel, sigma=organelle_seg_params.sigma, organelle_th=organelle_seg_params.organelle_th, 
        do_erode_dilate=organelle_seg_params.do_erode_dilate, do_remove_small_objects=organelle_seg_params.do_remove_small_objects, do_fill_holes=organelle_seg_params.do_fill_holes, 
        do_fill_holes_boarders=organelle_seg_params.do_fill_holes_boarders)  
        
        imgs1_seg_morph_slices[i] = imgs1_seg_morph_slices_
        imgs2_seg_morph_slices[i] = imgs2_seg_morph_slices_
    masks1 = (imgs1_seg_morph_slices//255).astype('uint8') # 0 or 1  (128, 16, 64, 64)
    masks2 = (imgs2_seg_morph_slices//255).astype('uint8') # 0 or 1  (128, 16, 64, 64)
    return masks1, masks2


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


def calculate_shortest_distance_between_objects_3d(labeled_array):
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
            
    avg_distance = np.array(list(distances.values())).mean()
   
    return avg_distance # scalar of average distances


def do_slice_markers( img3D ): # input tensor (16,64,64) 0-1 # nuclioli th 65
    slices_markers = []
    for i in range(img3D.shape[0]):
        ret, slice_markers = cv2.connectedComponents(img3D[i])  # ret - number of objects found 
        updated_slice_markers = np.copy(slice_markers)
        for j in range( 1, len(np.unique(slice_markers)) ): # over markers in a single patch
            slice_single_obj = np.where(slice_markers == np.unique(slice_markers)[j], 1, 0) # 2D binary 0 or 1 for a single object
            if slice_single_obj.sum() < 36: # remove small segments
                updated_slice_markers[ slice_markers == np.unique(slice_markers)[j] ] = 0
        slices_markers.append( updated_slice_markers )
    slices_markers = np.array(slices_markers) # 16,64,64
    return slices_markers 


def make_same_marker3D(slices_markers, correct_seq_num):   ## (16, 64, 64) , 1
    ### make same objects in each slice get the same segment
    k = 10
    updated_slices_markers = [slices_markers[0]] # [(64,64)]
    for i in range(1, slices_markers.shape[0]): # over patches
        updated_slice_markers = np.copy(slices_markers[i]) # (64,64)
        for j in range( 1, len(np.unique(slices_markers[i])) ): # over markers in a single patch
            slice_single_obj = np.where(slices_markers[i] == np.unique(slices_markers[i])[j], 1, 0) # single binary obj 64x64
            obj_n = np.unique( slice_single_obj * updated_slices_markers[i-1] ) # unique integer     
            if len(obj_n) == 1: #  overlap with background --> new object imerged
                if slice_single_obj.sum() < 32:
                    updated_slice_markers[ slices_markers[i] == np.unique(slices_markers[i])[j] ] = 0
                else:
                    updated_slice_markers[ slices_markers[i] == np.unique(slices_markers[i])[j] ] = k  
                    k = k+1
            if len(obj_n) == 2:
                updated_slice_markers[ slices_markers[i] == np.unique(slices_markers[i])[j] ] = obj_n[-1]
            if len(obj_n) > 2:
                Npixel_overlap = [-1]
                for n in range(1,len(obj_n)):
                    Npixel_overlap.append( slices_markers[i][slices_markers[i]==obj_n[n]].sum() // obj_n[n] )
                updated_slice_markers[ slices_markers[i] == np.unique(slices_markers[i])[j] ] = obj_n[np.argmax(Npixel_overlap)]
                    
        updated_slices_markers.append(updated_slice_markers) 
    updated_slices_markers = np.array(updated_slices_markers)       
    ### [1,2,3,10,11] --> [1,2,3,4,5]
    slices_markers_new = np.copy(updated_slices_markers)
    for i in range( len(np.unique(updated_slices_markers)) ):
        slices_markers_new[updated_slices_markers==np.unique(updated_slices_markers)[i]] = i
    updated_slices_markers = np.copy(slices_markers_new)
    return updated_slices_markers

def watershed_on_seg3D(seg3D): # (16, 64, 64)
    # structure = np.ones((3, 3, 3), dtype=bool)
    # labeled_array, num_objects = label(seg3D, structure=structure)
    # return labeled_array
    ### or
    seg3D_markers = do_slice_markers(seg3D)
    seg3D_markers = make_same_marker3D(seg3D_markers, 1)  # (16, 64, 64) 
    return seg3D_markers


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
                intersection = np.logical_and(gt_labeled == gt_reg.label, 
                                              pred_labeled == pred_reg.label).sum()
                union = gt_reg.area + pred_reg.area - intersection
                iou_matrix[i, j] = intersection / union if union > 0 else 0

    # 3. Hungarian Matching (Optimal Assignment)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    hd_values = []
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        # Only calculate HD if there is a valid spatial match (IoU > 0)
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

    volume_diff = np.abs( np.array(imgs1_binary3D_volume).mean() - np.array(imgs2_binary3D_volume).mean() ) 
    cx_diff =     np.abs( np.array(imgs1_binary3D_cx).mean()     - np.array(imgs2_binary3D_cx).mean() )  
    cy_diff =     np.abs( np.array(imgs1_binary3D_cy).mean()     - np.array(imgs2_binary3D_cy).mean() )  
    cz_diff =     np.abs( np.array(imgs1_binary3D_cz).mean()     - np.array(imgs2_binary3D_cz).mean() )  
    cr_diff =     np.abs( np.array(imgs1_binary3D_cr).mean()     - np.array(imgs2_binary3D_cr).mean() )  
    
    avg_distance_1 = calculate_shortest_distance_between_objects_3d(mask3D1_markers)
    avg_distance_2 = calculate_shortest_distance_between_objects_3d(mask3D2_markers)
    
    # print(avg_distance_1, avg_distance_2)
    
    if np.isnan(avg_distance_1):
        avg_distance_1 = 0
    if np.isnan(avg_distance_2):
        avg_distance_2 = 0
    avg_distance = np.abs(avg_distance_1 - avg_distance_2)# np.abs(( (avg_distance_1+0.0001) / (avg_distance_2+0.0001) ) - 1)
    
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
