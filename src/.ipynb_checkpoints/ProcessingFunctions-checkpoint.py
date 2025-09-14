import skimage
import sklearn
import scipy
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile
import cv2 
import random

from DisplayFunctions import volumetric2sequence


def run_filter(img3D, filter_kernel = 5, sigma = 50, filter_type='median'): # should be 0-255 uint8
    img3D = img3D.astype('uint8') # 16,64,64
    ### denoising filter 
    img3D_denoised = []
    for i in range(img3D.shape[0]):
        if filter_type == 'median':
            # img3D_denoised.append( cv2.medianBlur( img3D[i,:,:]   , filter_kernel ) ) # 3-for nucleoli, NUcEnv
            img3D_denoised.append( scipy.ndimage.median_filter(img3D[i,:,:], size=filter_kernel) )
        elif filter_type == 'bilateral':
            img3D_denoised.append( cv2.bilateralFilter(img3D[i,:,:] , filter_kernel, sigma, sigma) ) 
    img3D_denoised = np.array(img3D_denoised) # 16,64,64
    return img3D_denoised.astype('uint8')

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

def fill_holes(binary3D, k=3):
    seg_morph_slices = []
    for i in range(binary3D.shape[0]):
        binary2D = binary3D[i,:,:].astype('uint8')
        seg_morph = scipy.ndimage.binary_fill_holes(binary2D , np.ones((k,k))).astype('uint8') # (doesnt fill holes touching boundaries) 
        seg_morph_slices.append(seg_morph.astype('uint8'))
    seg_morph_slices = np.array(seg_morph_slices)*255 # 16,64,64 0-1 --> 0-255
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

organelle_th = -2 # 'NucEnv' - 20  ,  'Nuclioli' - 65
filter_type = 'median' # 'bilateral' 'median'
filter_kernel = 3
sigma = 15 # 15
do_morph = False
k1,k2,k3 = 3,3,3

def segmentation_pipeline(images3D, filter_type, k1, k2, k3, filter_kernel, sigma, organelle_th, do_erode_dilate, do_remove_small_objects, do_fill_holes, do_fill_holes_boarders):
    GT_image_filter = run_filter((images3D*255), filter_kernel=filter_kernel, sigma=sigma, filter_type=filter_type) # 0-255
    seg_th, seg_stack = segment3D(GT_image_filter, organelle_th=organelle_th)  # 0-255
    if do_erode_dilate == True:
        seg_stack = erode_dilate(seg_stack, k=k1, iterations=1)                # 0-255
    if do_remove_small_objects == True:
        seg_stack = remove_small_objects_func(seg_stack, k=k2)                 # 0-255
    if do_fill_holes == True:
        seg_stack = fill_holes(seg_stack, k=k1)                                # 0-255
    if do_fill_holes_boarders == True:
        seg_stack = fill_holes_boarders(seg_stack, k=k3)                       # 0-255
    return seg_th, seg_stack

def slice_markers( img3D ): # input tensor (16,64,64) 0-1 # nuclioli th 65
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

def make_same_marker3D(slices_markers, correct_seq_num):    
    ### make same objects in each slice get the same segment
    k = 10
    updated_slices_markers = [slices_markers[0]]
    for i in range(1, slices_markers.shape[0]): # over patches
        updated_slice_markers = np.copy(slices_markers[i])
        for j in range( 1, len(np.unique(slices_markers[i])) ): # over markers in a single patch
            slice_single_obj = np.where(slices_markers[i] == np.unique(slices_markers[i])[j], 1, 0) # single binary obj
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

   
def make_same_marker_between_sequences(slices_markersA, slices_markersB): #  change B according to A    
    ### make same objects in each slice get the same segment
    k = 10
    updated_slices_markers = []
    for i in range(slices_markersB.shape[0]): # over patches
        updated_slice_markers = np.copy(slices_markersB[i])
        for j in range(1, len(np.unique(slices_markersB[i])) ): # over markers in a single patch
            slice_single_obj = np.where(slices_markersB[i] == np.unique(slices_markersB[i])[j], 1, 0) # single binary obj
            obj_n = np.unique( slice_single_obj * slices_markersA[i] ) # unique integer     
            if len(obj_n) == 1: #  overlap with background --> new object imerged
                if slice_single_obj.sum() < 32:
                    updated_slice_markers[ slices_markersB[i] == np.unique(slices_markersB[i])[j] ] = 0
                else:
                    updated_slice_markers[ slices_markersB[i] == np.unique(slices_markersB[i])[j] ] = k  
                    k = k+1
            if len(obj_n) == 2: # overlap with object with a certain number
                updated_slice_markers[ slices_markersB[i] == np.unique(slices_markersB[i])[j] ] = obj_n[-1]
            if len(obj_n) > 2:
                Npixel_overlap = [-1]
                for n in range(1,len(obj_n)):
                    Npixel_overlap.append( slices_markersB[i][slices_markersB[i]==obj_n[n]].sum() // obj_n[n] )
                updated_slice_markers[ slices_markersB[i] == np.unique(slices_markersB[i])[j] ] = obj_n[np.argmax(Npixel_overlap)]                    
        updated_slices_markers.append(updated_slice_markers) 
    updated_slices_markers = np.array(updated_slices_markers)    
    return updated_slices_markers

def make_contours_from_segmentation(binary3D): # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
    contours_slices_coor = []
    contours_slices_imgs = []
    for i in range(binary3D.shape[0]):
        full_contours_img = np.zeros((binary3D.shape[1], binary3D.shape[2]))
        cnts, h = cv2.findContours( binary3D[i].astype('uint8') , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL   cv2.RETR_TREE
        # print(h)
        # ggg
        cnts__ = []
        for j in range( len(cnts) ):
            if len(cnts[j]) > 30:
                cnts__.append(cnts[j])
                full_contours_img[cnts[j][:,0,1], cnts[j][:,0,0]] = 255 
        contours_slices_coor.append(cnts__)
        contours_slices_imgs.append(full_contours_img)               
    return contours_slices_coor, np.array(contours_slices_imgs)

def find_closest_distance_between_two_largest_contours_in_slice(slice_contours_coor, cntr1=0 , cntr2=1):
    contours_len = []
    first_largest_contour_img = np.zeros((64, 64))
    second_largest_contour_img = np.zeros((64, 64))
    for j in range( len(slice_contours_coor) ): 
        contours_len.append( slice_contours_coor[j].shape[0] )
    sorted_by_size = np.argsort(np.array(contours_len))[::-1] 
    first_largest_contour = slice_contours_coor[sorted_by_size[cntr1]] # (512,1,2)
    first_largest_contour_img[first_largest_contour[:,0,1], first_largest_contour[:,0,0]] = 255 
    second_largest_contour = slice_contours_coor[sorted_by_size[cntr2]]
    second_largest_contour_img[second_largest_contour[:,0,1], second_largest_contour[:,0,0]] = 255   
    
    min_distance = 100000
    for i in range( len(first_largest_contour) ): # run over largest contour
        for j in range( len(second_largest_contour) ): # # run over second largest contour
            distance = np.sqrt( (first_largest_contour[i,0,0]-second_largest_contour[j,0,0])**2 + (first_largest_contour[i,0,1]-second_largest_contour[j,0,1])**2  )
            if distance < min_distance:
                min_distance = distance
                min_distance_row1 = first_largest_contour[i,0,1]
                min_distance_row2 = second_largest_contour[j,0,1]
                min_distance_col1 = first_largest_contour[i,0,0]
                min_distance_col2 = second_largest_contour[j,0,0]
    return min_distance, min_distance_row1, min_distance_row2, min_distance_col1, min_distance_col2


def centerofmass_r(img3D):
    cx, cy, cz = scipy.ndimage.center_of_mass(img3D)
    com_r = np.sqrt(cx**2 + cy**2 + cz**2) 
    return com_r

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
        cx, cy, cz = scipy.ndimage.center_of_mass( single_obj_in_img3D )
        cr = np.sqrt(cx**2 + cy**2 + cz**2) 
        all_obj3D_volumes.append( volume ) 
        all_center_z.append(  cz )
        all_center_x.append(  cx )
        all_center_y.append(  cy )
        all_center_r.append(  cr )
    return all_obj3D_volumes, all_center_z, all_center_x, all_center_y, all_center_r



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


def create_random_3Dpatches(signals_data, targets_data, Npatches    , patch_sizeX            , patch_sizeY            , patch_sizeZ           , train=True):
    if train:
        random_ids = np.random.randint(len(signals_data), size=Npatches)
    else:
        np.random.seed(5)
        random_ids = np.random.randint(len(signals_data), size=Npatches)
    
    BFbatch = []
    FLbatch = []
    DNAbatch = []
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

def minmax_norm(imgs3D):
    return (imgs3D - imgs3D.min()) / (imgs3D.max() - imgs3D.min())