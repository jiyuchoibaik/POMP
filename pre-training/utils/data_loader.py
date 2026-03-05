# data_loader.py  ─ TCGA-LUAD (RNA-seq + WSI)
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
        self.x_rna          = data["x_rna"]          # list of lists
        self.wsi_paths      = data["wsi_paths"]       # list of npy paths
        self.max_num_region = max_num_region

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        # ── RNA-seq ──────────────────────────────────────────────────────────
        x_rna = torch.tensor(self.x_rna[index], dtype=torch.float32)

        # ── WSI 패치 ─────────────────────────────────────────────────────────
        npy_path = self.wsi_paths[index]
        regions  = np.load(npy_path)                  # (N, 3, 256, 256)

        if regions.shape[0] > self.max_num_region:
            # 랜덤 샘플링 (원본은 앞에서 자름 → 다양성을 위해 랜덤)
            idx     = np.random.choice(regions.shape[0],
                                       self.max_num_region, replace=False)
            regions = regions[idx]

        regions = torch.tensor(regions, dtype=torch.float32)

        return regions, x_rna   # (N, 3, 256, 256),  (n_genes,)


def build_dataset(pkl_path: str, max_num_region: int = 300) -> TCGALUADDataset:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"[Dataset] {len(data['case_ids'])}개 케이스, RNA 차원={data['n_genes']}")
    return TCGALUADDataset(data, max_num_region)