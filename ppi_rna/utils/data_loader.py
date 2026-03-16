# data_loader.py  ─ TCGA-LUAD (RNA-seq + WSI)
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class TCGALUADDataset(Dataset):
    """
    데이터 구조 (preprocess_rna.py + extract_patches.py 출력):
      data = {
        "case_ids":   [str, ...],          # 환자 ID 목록
        "x_rna":      [[float,...], ...],  # (N, n_genes) log1p 정규화된 RNA-seq
        "wsi_paths":  [str, ...],          # regions.npy 경로 목록
        "n_genes":    int,
      }
    """
    def __init__(self, data: dict, max_num_region: int = 300):
        self.case_ids       = data["case_ids"]
        self.x_rna          = data["x_rna"]          # (N, n_genes) list of lists or ndarray
        self.wsi_paths      = data["wsi_paths"]       # list of npy paths
        self.max_num_region = max_num_region

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        x_rna = torch.tensor(self.x_rna[index], dtype=torch.float32)

        npy_path = self.wsi_paths[index]
        if not npy_path or not os.path.exists(npy_path):
            raise FileNotFoundError(f"npy 파일 없음: '{npy_path}' (index={index})")

        regions = np.load(npy_path, allow_pickle=True)
        if regions.dtype == object:
            regions = regions.item()

        # 항상 max_num_region으로 맞춤 (부족하면 반복 샘플링)
        n = regions.shape[0]
        if n >= self.max_num_region:
            idx = np.random.choice(n, self.max_num_region, replace=False)
        else:
            idx = np.random.choice(n, self.max_num_region, replace=True)  # ← 부족하면 반복
        regions = regions[idx]  # 항상 (max_num_region, 3, 256, 256)

        regions = torch.tensor(regions, dtype=torch.float32)
        return regions, x_rna


def build_dataset(pkl_path: str, max_num_region: int = 300) -> TCGALUADDataset:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"[Dataset] {len(data['case_ids'])}개 케이스, RNA 차원={data['n_genes']}")
    return TCGALUADDataset(data, max_num_region)