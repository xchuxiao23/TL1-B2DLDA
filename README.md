# A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function

This is the official implementation of our paper "A Secure and Efficient Image Sharing Method Based on Bilateral Compressive Sensing with Multilevel Privacy Preserving Function". This research project is developed based on MATLAB, created by Chuxiao Xu.

## Repository Structure

#### `/data` Directory

- **`original/`**
  - Raw image datasets in .pgm and .bmp format:
    - `ORL/` (400 images, 40 subjects, 10 images/subject)
    - `Yale/` (165 images, 15 subjects, 11 images/subject)
- **`without_noise/`**
  - Preprocessed datasets in .mat format:
- **`noise/`**
  - Block, gaussian and salt pepper noisy variants in .mat format

#### **`/model` Directory**

- Trained projection matrices .mat format



## Acknowledgement

We gratefully acknowledge the open-source implementation of [2DLDA-T$\ell_1$](https://github.com/YangSkywalker/2DLDA-TL1), which provided foundational code structures for our T$\ell_1$-B2DLDA. 