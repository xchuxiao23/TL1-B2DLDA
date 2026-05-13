# A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function

This repository contains the official MATLAB implementation of the paper:
> W. Wu, C. Xu, D. Zhao, H. Peng, and F. Tong, "A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function," *IEEE Transactions on Information Forensics and Security*, doi: 10.1109/TIFS.2026.3692298.

The project was developed based on MATLAB by Chuxiao Xu.


## 📂 Project Structure

```text
.
├── 2DPG-Recon/                 # Image reconstruction algorithms for Level II users
│   ├── func_2DPG.m             # Core 2DPG reconstruction function
│   ├── OurMeasurementMatrix.m  # CS measurement matrix and masking key generation
│   ├── derivation_of_TV.m      # Total variation derivative calculation
│   └── RMS.m                   # Root mean square error calculation
│
├── data/                       # Preprocessed dataset files
│   ├── ORL32.mat / ORL128.mat  # ORL face database, 32x32 and 128x128
│   └── Yale32.mat / Yale128.mat # Yale face database, 32x32 and 128x128
│
├── TL1-B2DLDA/                 # Core training algorithm for Level I users
│   ├── TL1_B2DLDA.m            # Main TL1-B2DLDA optimization function
│   ├── TL1_B2DLDA_obj.m        # Objective function calculation
│   ├── Update_V.m / Update_W.m # Iterative projection matrix updates
│   └── ...
│
├── utils/                      # Utility functions
│   ├── ModDiffusion.m          # Secure transmission by bidirectional modulo diffusion
│   ├── Improved_HenonMap.m     # Chaotic map for measurement matrix generation
│   ├── Improved_ZeraouliaSprottMap.m # Chaotic map for privacy protection
│   ├── knn_classifier2D_gpu.m  # GPU-accelerated KNN classifier
│   └── randomSplit2D.m         # Dataset splitter
│
├── WaveletSoftware/            # Third-party wavelet transform dependencies
├── train_TL1_B2DLDA.m          # Entry point for training projection matrices W and V
└── Transmission_Pipline.m      # Entry point for the full sampling, encryption, and reconstruction pipeline
```

## 🚀 Usage

### Training for Level I Privacy

1. Open `train_TL1_B2DLDA.m`.
2. Set the dataset path, such as `./data/ORL128.mat`, and configure the parameters.
3. Run the script. The trained matrices are saved to the `checkpoint/` directory, which is created automatically.

### Transmission Pipeline Simulation

1. Open `Transmission_Pipline.m`.
2. Make sure trained matrices are available from the training step, or update the script to load existing checkpoints.
3. Run the script to perform CS sampling, masking, restoration, decryption, and image reconstruction.
4. The script displays the original and reconstructed images and reports the PSNR.

## ⚙️ Requirements

- MATLAB R2020a or later is recommended.
- Parallel Computing Toolbox is required for GPU data structures.
- An NVIDIA GPU with CUDA compute capability 3.5 or higher is recommended.

The core `TL1-B2DLDA` algorithm uses `gpuArray` for acceleration. Running the algorithm on CPU requires removing or replacing the GPU-dependent code paths.

## 📢 Dataset Availability

This repository includes preprocessed ORL and Yale datasets in `.mat` format for quick testing.

Due to GitHub file size limitations, the Caltech-101 and FEI Face Database used in the paper's large-scale experiments are not included in this repository. Researchers interested in reproducing the generalization experiments on these larger datasets can contact the author at <xchuxiao23@bupt.edu.cn>.

## 🙏 Acknowledgement

We gratefully acknowledge the open-source implementation of [2DLDA-TL1](https://github.com/YangSkywalker/2DLDA-TL1), which provided foundational code structures for TL1-B2DLDA.

We also thank [2DCS-ETC](https://github.com/zhangboswjtu/2DCS-ETC) for providing the reference implementation for the 2D compressed sensing reconstruction code.


## 📖 Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{wu2026secure,
  author  = {Wu, W. and Xu, C. and Zhao, D. and Peng, H. and Tong, F.},
  title   = {A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  doi     = {10.1109/TIFS.2026.3692298}
}
```
