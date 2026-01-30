# A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function

This is the official implementation of our paper **"A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function"**. 

This research project is developed based on MATLAB, created by Chuxiao Xu.

> **Note:** This paper has been submitted to **IEEE Transactions on Information Forensics and Security (TIFS)**.



## 📂 Project Structure

The project is organized as follows:

```text
.
├── 2DPG-Recon/                 # Algorithms for Image Reconstruction (Level II User)
│   ├── func_2DPG.m             # Core 2DPG reconstruction function
│   ├── OurMeasurementMatrix.m  # Generates CS measurement matrices & Masking keys
│   ├── derivation_of_TV.m      # Total Variation derivative calculation
│   └── RMS.m                   # Root Mean Square error calculation
│
├── data/                       # Dataset files
│   ├── ORL32.mat / ORL128.mat  # ORL Face Database (32x32 / 128x128)
│   └── Yale32.mat / Yale128.mat # Yale Face Database
│
├── TL1-B2DLDA/                 # Core Training Algorithm (Level I User)
│   ├── TL1_B2DLDA.m            # Main function for Tl1-B2DLDA optimization
│   ├── TL1_B2DLDA_obj.m        # Objective function calculation
│   ├── Update_V.m / Update_W.m # Iterative update functions for projection matrices
│   └── ...
│
├── utils/                      # Utility Functions
│   ├── ModDiffusion.m          # Secure Transmission: Bidirectional Modulo Diffusion
│   ├── Improved_HenonMap.m     # Chaotic map for measurement matrix generation
│   ├── Improved_ZeraouliaSprottMap.m # Chaotic map for privacy protection
│   ├── knn_classifier2D_gpu.m  # GPU-accelerated KNN classifier
│   └── randomSplit2D.m         # Dataset splitter
│
├── WaveletSoftware/            # Third-party dependencies for Wavelet transform
│
├── train_TL1_B2DLDA.m          # [Entry Point] Script to train projection matrices (W, V)
└── Transmission_Pipline.m      # [Entry Point] Full pipeline simulation (Sampling -> Encryption -> Reconstruction)
```



##  🚀 Usage

### 1. Training (Level I Privacy)

To train the projection matrices ($W$ and $V$) and evaluate the classification performance:

1. Open `train_TL1_B2DLDA.m`.
2. Set the dataset path (e.g., `./data/ORL128.mat`) and parameters.
3. Run the script. It will save the trained matrices to the `checkpoint/` directory (created automatically).

### 2. Transmission Pipeline Simulation

To simulate the complete secure image sharing process, including CS sampling, encryption (masking), and Level II reconstruction:

1. Open `Transmission_Pipline.m`.
2. Ensure you have trained matrices (from step 1) or point to existing checkpoints.
3. Run the script. It will:
   - Perform CS sampling.
   - Masking and Restoration
   - Decrypt and reconstruct the image.
   - Display the **Original vs. Reconstructed** comparison and calculate PSNR.



## ⚙️ Requirements

To run this code, you need the following environment:

- **MATLAB** (R2020a or later recommended).
- **Parallel Computing Toolbox** (Required for GPU data structures).
- **Hardware**: An **NVIDIA GPU** with CUDA compute capability 3.5 or higher.
  - *Note: The core algorithm `TL1-B2DLDA` is optimized using `gpuArray` for acceleration. Running on CPU requires modifying the code to remove GPU dependencies.*



## 📢 Dataset Availability

This repository includes the preprocessed **ORL** and **Yale** datasets (in `.mat` format) for quick testing.

Due to GitHub's file size limitations, the **Caltech-101** and **FEI Face Database** used in the paper's large-scale experiments are **not included** in this repository.

Researchers interested in reproducing the generalization experiments on these larger datasets can contact the author at: 📧 **xchuxiao23@bupt.edu.cn**



## 🙏 Acknowledgement

We gratefully acknowledge the open-source implementation of [2DLDA-TL1](https://github.com/YangSkywalker/2DLDA-TL1), which provided foundational code structures for our TL1-B2DLDA. We also thank [2DCS-ETC](https://github.com/zhangboswjtu/2DCS-ETC) for providing the reference implementation for the 2D Compressed Sensing reconstruction code.

