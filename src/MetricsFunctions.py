import skimage
import sklearn
from sklearn.metrics import normalized_mutual_info_score
import scipy
from scipy import ndimage
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from glrlm import GLRLM
import torch
# import torch.nn.functional as F
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from src import resnet
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2 
import random
from generative.metrics import MultiScaleSSIMMetric

# import sys
# sys.path.append(".")
# main_path = "/.../" ## change to main directory of github
# sys.path.append(main_path+'src')

#########################################################


def lpips(main_path, imgs1, imgs2):
    device = torch.device("cuda") 
    CLS_model = resnet.generate_model(
    model_depth=101,
    n_classes=6,
    n_input_channels=1,
    shortcut_type='B',
    conv1_t_size=7,
    conv1_t_stride=1,
    no_max_pool=False,
    widen_factor=1.0)

    CLS_model.load_state_dict(torch.load(main_path + "saved_models/CLS/CLS.pth"))  
    CLS_model.to(device)
    CLS_model.eval()

    return_layers = {
        'layer1': 'layer1_output',  
        'layer2': 'layer2_output', 
        'layer3': 'layer3_output',  
        'layer4': 'layer4_output',
        'fc': 'layerFC_output',}

    layer_losses = torch.zeros((1, imgs1.shape[0])).to(device)
    mid_getter = MidGetter(CLS_model, return_layers)
    prediction_intermidiate_outputs = mid_getter(imgs1)
    GT_intermidiate_outputs = mid_getter(imgs2) 
    for i in range(len(list(return_layers.keys()))):    
        GT_layer_features   = GT_intermidiate_outputs[0][list(return_layers.values())[i]]
        pred_layer_features = prediction_intermidiate_outputs[0][list(return_layers.values())[i]]
        layer_loss = torch.abs( GT_layer_features  -  pred_layer_features )
        layer_loss = layer_loss**2
        try:
            layer_loss = layer_loss.mean(axis=(1,2,3,4)) ## torch.Size([bs]) 
        except:
            layer_loss = layer_loss.mean(axis=(1)) ## torch.Size([bs]) 
        layer_losses = torch.concatenate((layer_losses, layer_loss.unsqueeze(0)), axis=0)
    layer_losses = layer_losses[1:, :]

    return layer_losses.cpu().detach().numpy()


def calc_MI(imgsA, imgsB, bins):
    hgram = np.histogram2d(imgsA.ravel(), imgsB.ravel(), bins)[0]
    g, p, dof, expected = chi2_contingency(hgram, lambda_="log-likelihood")
    mi = 0.5 * g / (hgram.sum() + 0.00001)
    return mi

def calc_nmi(imgsA, imgsB,bins): ## implementation of sklearn.metrics.normalized_mutual_info_score
    eps = 0.0000000001
    arr1 = (( imgsA - imgsA.min() ) / ( imgsA.max() - imgsA.min() )*(bins-1)).astype('uint8')
    arr2 = (( imgsB - imgsB.min() ) / ( imgsB.max() - imgsB.min() )*(bins-1)).astype('uint8')

    arr1 = arr1.astype('uint8')
    arr2 = arr2.astype('uint8')
    joint_prob_mat = np.zeros((bins,bins))
    for i in range(len(arr1)):
        joint_prob_mat[arr1[i], arr2[i]] += 1
    joint_prob_mat = joint_prob_mat / (np.sum(joint_prob_mat)+eps)
    marg_prob_arr1 = np.sum(joint_prob_mat, axis=1) # (bins)
    marg_prob_arr2 = np.sum(joint_prob_mat, axis=0) # (bins)
    entropy_arr1 = -np.sum( marg_prob_arr1*np.log2(marg_prob_arr1+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=1) * np.log2(joint_prob_mat.sum(axis=1)+eps) )
    entropy_arr2 = -np.sum( marg_prob_arr2*np.log2(marg_prob_arr2+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=0) * np.log2(joint_prob_mat.sum(axis=0)+eps) )
    joint_entropy = -np.sum(joint_prob_mat * np.log2(joint_prob_mat+eps)) 
    mi = entropy_arr1 + entropy_arr2 - joint_entropy
    # mi = 0
    # for i in range(bins):
    #     for j in range(bins):
    #         if joint_prob_mat[i,j] > 0:
    #             mi = mi + joint_prob_mat[i,j] * np.log2( joint_prob_mat[i,j] / (marg_prob_arr1[i]*marg_prob_arr2[j]) + eps)
    # nmi = 2*mi / ( entropy_arr1 + entropy_arr2 + eps)
    return mi

def calc_sklearn_nmi(imgsA, imgsB,bins):
    arr1 = (( imgsA - imgsA.min() ) / ( imgsA.max() - imgsA.min() )*(bins-1)).astype('uint8')
    arr2 = (( imgsB - imgsB.min() ) / ( imgsB.max() - imgsB.min() )*(bins-1)).astype('uint8')
    return normalized_mutual_info_score(arr1.ravel(), arr2.ravel())

def entropy(img3D, bins=20):
    eps = 0.000000000001
    hgram, x_edges, y_edges = np.histogram2d( img3D.ravel(),img3D.ravel(), bins=bins)
    px = np.sum(hgram / float(np.sum(hgram)), axis=1) # marginal for x over y
    Hx = -np.sum( px*np.log2(px+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=1) * np.log2(joint_prob_mat.sum(axis=1)+eps) )
    return Hx

def MI(imgsA, imgsB, bins=20):
    eps = 0.000000000001
    hgram, x_edges, y_edges = np.histogram2d( imgsA.ravel(),imgsB.ravel(), bins=bins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    Hx = -np.sum( px*np.log2(px+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=1) * np.log2(joint_prob_mat.sum(axis=1)+eps) )
    Hy = -np.sum( py*np.log2(py+eps) ) # scalar  -np.sum( joint_prob_mat.sum(axis=0) * np.log2(joint_prob_mat.sum(axis=0)+eps) )
    Hxy = -np.sum(pxy * np.log2(pxy+eps)) 
    mi = Hx + Hy - Hxy
    return mi

# Generate GLCM for texture properties
def glcm_measurements(img3D, distances, angles): # distance - Offset between pixelsb, angle = Vertical Direction  0°, 45°, 90°-np.pi/2, and 135°
    glcm = np.zeros((256,256,1,1))
    for slice in range(img3D.shape[0]): # over slices (16, 64, 64)
        glcm = glcm + skimage.feature.graycomatrix( (img3D[slice]*255).astype('uint8'), distances=distances, angles=angles,levels=256 ) # 256,256,1,1
    # Calculate Features from GLCM
    contrast      = skimage.feature.graycoprops(glcm, 'contrast')[0][0]
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity   = skimage.feature.graycoprops(glcm, 'homogeneity')[0][0]
    energy        = skimage.feature.graycoprops(glcm, 'energy')[0][0]
    correlation   = skimage.feature.graycoprops(glcm, 'correlation')[0][0]
    return correlation

def glrlm_measurements(img3D, Pixel_norm_level): # distance - Offset between pixelsb, angle = Vertical Direction  0°, 45°, 90°-np.pi/2, and 135°
    glrlm = np.array([0,0,0,0,0])
    for slice in range(img3D.shape[0]): # over slices (16, 64, 64)
        glrlm = glrlm + np.array(GLRLM().get_features(img3D[slice], Pixel_norm_level).Features)
    glrlm = glrlm / img3D.shape[0]
    sre, lre, glu, rlu, rpc = glrlm
    return sre, lre, glu, rlu, rpc  

def gabor_skimage(img3D):
    gabors2D = []
    for slice in range(img3D.shape[0]):
        img2D = (img3D[slice] - img3D[slice].mean()) / img3D[slice].std()
        for theta in (0, 1):
            theta = theta / 4. * np.pi
            for frequency in (0.1, 0.4):
                kernel = gabor_kernel(frequency, theta=theta)
                gabors2D.append(  np.sqrt(ndimage.convolve(img2D, np.real(kernel), mode='wrap')**2 + ndimage.convolve(img2D, np.imag(kernel), mode='wrap')**2)  )  
    return np.array(gabors2D)

def ms_ssim(imgs1, imgs2, rn=3):
    return np.round( MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)( torch.tensor(imgs1), torch.tensor(imgs2) ).mean().numpy(), rn)

def pcc(imgs1, imgs2, rn=3):
    return np.round( stats.pearsonr( imgs1.flatten() , imgs2.flatten() )[0], rn)

def mse(imgs1, imgs2, rn=3):
    return np.round( np.mean(np.abs( imgs1.flatten() - imgs2.flatten() )**2), rn)

def jsd(imgs1, imgs2, rn=3):
    return np.round( jensenshannon( imgs1.flatten(), imgs2.flatten(), base=2 ), rn)

def mi(imgs1, imgs2, rn=3):
    return np.round( MI(imgs1, imgs2, bins=20), rn) 

