# DM4ISL (Diffusion model for in silico labeling)
Oded Rotem, Orit Kliper-Gross, Assaf Zaritsky


## Paper Abstract

In silico labeling predicts organelle-specific localizations from label-free images, potentially enabling longitudinal and multiplexed live-cell imaging. Here we present DM4ISL (Diffusion Model for In Silico Labeling), a high-fidelity and uncertainty-aware model that overcomes current limitations, such as blurry or structurally inaccurate predictions, by resolving the intrinsic entanglement of biological signal and photon noise that characterizes fluorescence imaging. By leveraging the iterative nature of diffusion, DM4ISL employs a step-wise self-ensemble averaging approach that marginalizes these stochastic fluctuations, preventing the overfitting to noise that typically degrades standard final-step predictions. Benchmarking across six organelles demonstrates that DM4ISL systematically outperforms state-of-the-art architectures in structural fidelity, as confirmed by both pixel-based and application-specific metrics. By using the standard deviation across intermediate denoising steps, DM4ISL generates localized uncertainty maps that automatically detect erroneous predictions at inference time. By providing high-fidelity predictions alongside built-in reliability metrics, DM4ISL establishes a new standard for trustworthy, data-driven discovery in cell biology.


## Framework
![MainFigure](https://github.com/zaritskylab/DM4ISL/blob/main/figures/MainFigure.png)  


### Overview
DM4ISL is a framework designed to predict organelle fluorescence in label-free microscopy images based on diffusion models and at inference optimized mechanism for improved results. This repository includes the source code for training, inference, and results analysis.

### Data

The full data of paired brightfield (BF) and fluorescence (FL) images can be downloaded from the allen intitute of cell science:
https://open.quiltdata.com/b/allencell/tree/aics/pipeline_integrated_cell/

To download the data in the same data-structure used in this work so that results can be replicated, use the code to download the data from:
https://github.com/zaritskylab/MaskInterpreter/blob/main/md/data.md

In the paper, we trained separate models for 6 different organelles (DNA, nuclear envelope, nucleoli, actin filament, mitochondria and microtubules) from the link.  
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
    conda create -n DM4ISL_env python=3.9.23
    conda activate DM4ISL_env
    pip install -r requirements.txt
    pip install notebook
    
3. A trained model for Nuclear envelope can be downloaded from hugging face [https://huggingface.co/OdedRot/DM4ISL/tree/main](https://huggingface.co/OdedRot/DDPM4ISL/tree/main)
   
   Add NucEnv.pth to saved_models/NucEnv   -   This is a trained DM4ISL model for Nuclear envelope organelle.
5. For every notebook you use, update the "main_path" directory path.
6. To train a new model use the TRAIN notebook. Train for ~40 epochs and 100 steps per epoch.
7. To run inference on a trained model use the INFERENCE notebook.
8. To evaluate predictions on a trained model use the RESULTS_ANALYSIS notebook.
9. Our work was done on RTX6000. Use with similar GPU capabilities.

## Acknowledgement
Our work builds upon the diffusion model MONAI framework. We adapted the "image_to_image_translation" for 3D images and for our DM4ISL inference process.
[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/GenerativeModels/tree/main/tutorials/generative)


