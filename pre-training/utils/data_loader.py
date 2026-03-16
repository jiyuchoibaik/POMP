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
        "x_rna":      [[float,...], ...],  # (N, n_genes) 정규화된 RNA (log1p 또는 z-score)
        "wsi_paths":  [str, ...],          # regions.npy 경로 목록 (pretrain 시 필수)
        "n_genes":    int,
        # z-score pkl 추가 키: "hvg_genes_mean", "hvg_genes_std" (fine-tuning 시 동일 정규화용)
      }
    """
    def __init__(self, data: dict, max_num_region: int = 300):
        case_ids = data["case_ids"]
        x_rna = data["x_rna"]
        wsi_paths = data["wsi_paths"]
        # regions.npy 없는 케이스는 제외 (하나 빼고 학습)
        self.case_ids = []
        self.x_rna = []
        self.wsi_paths = []
        skipped = []
        for i, path in enumerate(wsi_paths):
            if path and os.path.exists(path):
                self.case_ids.append(case_ids[i])
                self.x_rna.append(x_rna[i])
                self.wsi_paths.append(path)
            else:
                skipped.append((i, case_ids[i] if i < len(case_ids) else "?", path or "(empty)"))
        self.max_num_region = max_num_region
        if skipped:
            print(f"[Dataset] regions.npy 없음 {len(skipped)}건 제외 → 학습 샘플 {len(self.case_ids)}개")
            for idx, cid, p in skipped[:5]:
                print(f"  제외: index={idx} case_id={cid} path={p}")
            if len(skipped) > 5:
                print(f"  ... 외 {len(skipped) - 5}건")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        x_rna = torch.tensor(self.x_rna[index], dtype=torch.float32)

        npy_path = self.wsi_paths[index]

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


def _resolve_wsi_path(path: str, root: str) -> str:
    """상대경로를 root 기준 절대경로로. 없으면 downloads/wsi → patches 로 대체 시도."""
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(root, path))
    if os.path.exists(path):
        return path
    # regions.npy가 실제로는 datasets/patches/ 에 있는 경우 (pkl은 downloads/wsi 로 저장된 경우)
    for old, new in [("downloads/wsi", "patches"), ("downloads\\wsi", "patches")]:
        if old in path:
            fallback = path.replace(old, new)
            if os.path.exists(fallback):
                return fallback
            break
    return path


def build_dataset(pkl_path: str, max_num_region: int = 300) -> TCGALUADDataset:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pkl_abs = os.path.abspath(pkl_path)
    root = os.path.dirname(os.path.dirname(pkl_abs))  # .../datasets/x.pkl -> ... (pre-training)
    wsi_paths = data["wsi_paths"]
    resolved = [_resolve_wsi_path(p, root) for p in wsi_paths]
    data = {**data, "wsi_paths": resolved}
    print(f"[Dataset] {len(data['case_ids'])}개 케이스, RNA 차원={data['n_genes']}")
    return TCGALUADDataset(data, max_num_region)