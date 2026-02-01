# DM4ISL (Diffusion model for in silico labeling)
Oded Rotem, Assaf Zaritsky


## Paper Abstract

Fluorescence microscopy is a useful tool for cell and subcellular quantitative analysis, driving drug discovery and biological research. Label-free fluorescence prediction, also known as in-silico labeling, is a rapidly advancing field that seeks to digitally infer fluorescence microscopy images from label-free modalities such as brightfield. This approach has the potential to overcome key limitations of fluorescence labeling, including phototoxicity, spectral overlap, and limited multiplexing. In this work, we test the efficacy of applying Denoising Diffusion Probabilistic Models (DDPM). We systematically compare the DDPM predictions to those from conventional models such as U-Net and GAN across six organelle classes using volumetric (z-stack) fluorescence-brightfield paired data from the Allen Cell Institute. Our approach introduces DDPMavg, an averaged output over intermediate inference steps, which achieves consistently higher similarity to ground truth and significantly fewer prediction artifacts. We evaluated our method performance across 14 quantitative metrics, including pixel-wise intensity matching, structural integrity, texture preservation, and biological measurement consistency. Beyond prediction capabilities, we leverage the generative nature of diffusion models to estimate prediction uncertainty through both intra- and inter-process variance. This enables the identification of erroneous artifacts such as hallucinated or deformed structures, a previously under-addressed concern in label-free prediction. Our results demonstrate a 17.9% average improvement with fewer artifacts. We also show DDPM generates fewer erroneous structures compared to Unet and our uncertainty mechanism captures all erroneous instances which were annotated manually. Providing state-of-the-art predictions for complex cellular substructures while also introducing a valuable confidence mechanism, pushes label-free prediction one step closer to practical, trustworthy deployment in biological discovery.

## Framework
![MainFigure](https://github.com/user-attachments/assets/30d3b6d9-9fae-4c58-ae02-a0c638540882](https://github.com/zaritskylab/DM4ISL/blob/main/figures/MainFigure.png)  


### Overview
DM4ISL is a framework designed to predict organelle fluorescence in label-free microscopy images based on diffusion models and at inference optimized mechanism for improved results. This repository includes the source code for training, inference, and results analysis.

### Data

The full data of paired brightfield and fluorescence images can be downloaded from the allen intitute of cell science
[https://open.quiltdata.com/b/allencell/packages/aics/label-free-imaging-collection/tree/latest/](https://open.quiltdata.com/b/allencell/packages/aics/label-free-imaging-collection/tree/latest/)

We provide several patches of Nuclear Envelope under data/NucEnv, allowing you to run the training, inference and analysis notebooks.

**The data directory includes**:
- BF - brightfield patches
- GT - fluorescence patches use for ground truth

**The INFERENCE notebook will save data into the following folders:**
- FL_pred - predictions of the DDPM final output
- FLavg_pred - predictions of the DM4ISL average
- FLavg_std - standard deviation images using the intermidiate timesteps
- FLavg_std_seg - binary image of FLavg_std spotting erroneuos locations
- FLavg_std_seeds - standard deviation images using the intermidiate timesteps initiaed from multiple seeds
- FLavg_std_seg_seeds - binary image of FLavg_std_seeds spotting erroneuos locations

**To run the RESULTS_ANALYSIS notebook we also provide samples from Unet and GAN predictions:**
- Unet
- GAN


### Example Notebooks
- **TRAINING notebook**: 

    This notebook demonstrates how to train the DM4ISL model using paired brightfield and organelle fluoresence images. 
    
- **INFERENCE notebook**:

    This notebook demonstrates how to predict a flouresence image using a trained DM4ISL model and brightfield input images. 

- **RESULTS_ANALYSIS notebook**:

    This notebook demonstrates how to analyze prediction results using all evaluation metrics described in the paper.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/zaritskylab/DM4ISL

2. Use python==3.9 and install the required dependencies:
    ```bash
    %cd DM4ISL
    conda create -n DM4ISL_env python=3.9
    conda activate DM4ISL_env
    pip install -r requirements.txt
    pip install notebook
    
3. Download two models from hugging face https://huggingface.co/OdedRot/DM4ISL/tree/main
   
   Add CLS.pth to saved_models/CLS
   
   Add NucEnv.pth to saved_models/NucEnv
5. For every notebook you use, update the "main_path" directory path.
6. To train a new model use the TRAIN notebook.
7. To run inference on a trained model use the INFERENCE notebook.
8. To evaluate predictions on a trained model use the RESULTS_ANALYSIS notebook.

## Acknowledgement
We used MONAI framework found here:
https://github.com/Project-MONAI/MONAI

