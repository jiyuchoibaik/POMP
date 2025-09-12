# POMP: Pathology-omics Multimodal Pre-training Framework for Cancer Survival Prediction
![image](https://github.com/SuixueWang/POMP/POMP-framework.png)

This is a PyTorch implementation of the POMP paper under Linux with GPU NVIDIA A100 80GB.

## Requirements
- pytorch==1.8.0+cu111
- torchvision==0.9.0+cu111
- torchaudio==0.8.0
- Pillow==9.5.0
- timm==0.3.2
- lifelines==0.27.4

## Preprocessing
- The preprocessed multi-omics data are stored as pickle files in the `pre-training/datasets/` and `survival/datasets/` directories, ready for direct use.
- Due to the large size of the whole-slide pathology images, users need to download them manually from the TCGA portal. The procedure is as follows:
   - (1) Extract the pathology image IDs using the information from the `'region_pixel_5x'` field in the provided pickle files.
   - (2) Download the corresponding whole-slide images from the TCGA portal based on the extracted IDs.
   - (3) Perform image patching using the method described in the paper.



## Pre-training
```angular2htm
  CUDA_VISIBLE_DEVICES=0 python3 pre-training/main_multimodal_pretrain.py
```

## Survival prediction
```angular2html
  CUDA_VISIBLE_DEVICES=0 python3 survival/main_multimodal_survival.py
```
