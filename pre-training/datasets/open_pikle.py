import pickle
import os


with open("/Users/choijiyubaik/Documents/DAC/26춘계_의정학/POMP/pre-training/datasets/datasets_multimodal_pretrain.pkl", "rb") as f:

    data = pickle.load(f)

paths = data["region_pixel_5x"]

# 전체 항목 확인 (ellipsis 없이)
with open("wsi_files.txt", "w") as f:
    luad_paths = [p for p in paths if "TCGA-LUAD" in p]

    for p in luad_paths:
        f.write(p+"\n")

