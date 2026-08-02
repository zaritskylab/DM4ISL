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
from scipy.ndimage import label
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


