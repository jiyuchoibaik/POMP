#!/usr/bin/env python3
"""
사전 학습된 모델의 이미지–RNA 정렬 품질 평가.

- 평가 1: Retrieval Accuracy (Image→RNA, RNA→Image, Top-1/5/10)
- 평가 2: t-SNE 시각화 (img_cls 파란색, omics_cls 빨간색, 동일 환자 쌍 선 연결)

사용: pre-training 디렉터리에서
  python scripts/eval_alignment.py \
    --checkpoint ../output_pretrain_zscore_2/checkpoint-tcga_luad_v2-500.pth \
    --pkl ./datasets/rna_processed_zscore.pkl \
    --out ./output_pretrain_zscore_2/alignment_eval.png
"""
import argparse
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

# pre-training 루트를 path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PRETRAIN_ROOT)
os.chdir(PRETRAIN_ROOT)

from model.models_pomp import vit_base_patch16


def _resolve_wsi_path(path: str, root: str) -> str:
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(root, path))
    if os.path.exists(path):
        return path
    for old, new in [("downloads/wsi", "patches"), ("downloads\\wsi", "patches")]:
        if old in path:
            fallback = path.replace(old, new)
            if os.path.exists(fallback):
                return fallback
            break
    return path


class EvalDataset(torch.utils.data.Dataset):
    """재현 가능하도록 region을 결정론적으로 선택 (첫 N개 또는 반복)."""
    def __init__(self, data: dict, pkl_path: str, max_num_region: int = 300):
        self.case_ids = list(data["case_ids"])
        self.x_rna = list(data["x_rna"])
        pkl_abs = os.path.abspath(pkl_path)
        root = os.path.dirname(os.path.dirname(pkl_abs))  # .../datasets/x.pkl → .../pre-training
        self.wsi_paths = [_resolve_wsi_path(p, root) for p in data["wsi_paths"]]
        self.max_num_region = max_num_region
        # regions.npy 없는 케이스 제외
        valid = []
        for i, p in enumerate(self.wsi_paths):
            if p and os.path.exists(p):
                valid.append(i)
        self.valid_idx = valid
        self.case_ids = [self.case_ids[i] for i in valid]
        self.x_rna = [self.x_rna[i] for i in valid]
        self.wsi_paths = [self.wsi_paths[i] for i in valid]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        x_rna = torch.tensor(self.x_rna[index], dtype=torch.float32)
        regions = np.load(self.wsi_paths[index], allow_pickle=True)
        if regions.dtype == object:
            regions = regions.item()
        n = regions.shape[0]
        # 결정론적: 앞에서부터 채우고 부족하면 반복
        if n >= self.max_num_region:
            idx = np.arange(self.max_num_region)
        else:
            idx = np.tile(np.arange(n), (self.max_num_region // n + 1))[:self.max_num_region]
        regions = torch.tensor(regions[idx], dtype=torch.float32)
        return regions, x_rna


def load_data_and_embeddings(checkpoint_path: str, pkl_path: str, device: str, max_patches: int = 300, n_genes: int = None):
    """체크포인트·pkl 로드 후 전체 환자에 대해 img_cls, omics_cls 추출 (배치 1)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if n_genes is None:
        n_genes = data.get("n_genes", 2000)
    dataset = EvalDataset(data, pkl_path, max_num_region=max_patches)
    n_samples = len(dataset)
    if n_samples == 0:
        raise SystemExit("유효한 샘플이 없습니다 (regions.npy 경로 확인).")
    print(f"[Eval] 유효 샘플 수: {n_samples}, n_genes: {n_genes}")

    model = vit_base_patch16(rna_dim=n_genes, drop_path_rate=0.0, global_pool=True, num_classes=0)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    img_cls_list = []
    omics_cls_list = []
    with torch.no_grad():
        for regions, x_rna in tqdm(loader, desc="Embedding", unit="sample", total=len(dataset)):
            regions = regions.to(device)
            x_rna = x_rna.to(device)
            img_cls, omics_cls, _, _ = model([regions, x_rna])
            img_cls_list.append(img_cls.cpu().numpy())
            omics_cls_list.append(omics_cls.cpu().numpy())
    img_cls = np.concatenate(img_cls_list, axis=0)   # (N, 1, D)
    omics_cls = np.concatenate(omics_cls_list, axis=0)  # (N, 1, D)
    return img_cls, omics_cls, n_samples


def retrieval_accuracy(img_cls: np.ndarray, omics_cls: np.ndarray, k_list=(1, 5, 10)):
    """Image→RNA, RNA→Image Top-k 정확도."""
    N = img_cls.shape[0]
    img = img_cls.reshape(N, -1).astype(np.float64)
    omics = omics_cls.reshape(N, -1).astype(np.float64)
    # 정규화
    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
    omics = omics / (np.linalg.norm(omics, axis=1, keepdims=True) + 1e-8)
    sim_img_to_omics = img @ omics.T   # (N, N)
    sim_omics_to_img = omics @ img.T   # (N, N)

    results = {}
    for direction, sim in [("Image→RNA", sim_img_to_omics), ("RNA→Image", sim_omics_to_img)]:
        # 각 행에서 자신 인덱스 제외하고 상대와의 유사도만 사용 (자기 자신이 1이면 항상 1등)
        for k in k_list:
            # top-k 인덱스 (각 행 기준)
            topk = np.argsort(-sim, axis=1)[:, :k]   # (N, k)
            hit = np.any(topk == np.arange(N)[:, None], axis=1)
            acc = hit.mean() * 100.0
            results[f"{direction} Top-{k}"] = acc
    return results


def tsne_visualization(img_cls: np.ndarray, omics_cls: np.ndarray, out_path: str):
    """746개 벡터 (373 img + 373 omics) t-SNE → 2D, 파란/빨간 점 + 동일 환자 쌍 선."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"t-SNE/시각화 스킵 (필요: pip install scikit-learn matplotlib): {e}")
        return
    N = img_cls.shape[0]
    img = img_cls.reshape(N, -1)
    omics = omics_cls.reshape(N, -1)
    all_vec = np.concatenate([img, omics], axis=0)   # (746, D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, all_vec.shape[0] // 10)))
    xy = tsne.fit_transform(all_vec)
    img_xy = xy[:N]
    omics_xy = xy[N:]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(img_xy[:, 0], img_xy[:, 1], c="blue", s=20, alpha=0.7, label="WSI CLS")
    ax.scatter(omics_xy[:, 0], omics_xy[:, 1], c="red", s=20, alpha=0.7, label="RNA CLS")
    for i in range(N):
        ax.plot([img_xy[i, 0], omics_xy[i, 0]], [img_xy[i, 1], omics_xy[i, 1]], "k-", alpha=0.15, linewidth=0.5)
    ax.legend()
    ax.set_title("Pre-training alignment: Image (blue) vs RNA (red), same patient connected")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE 저장: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="사전 학습 정렬 품질 평가 (Retrieval + t-SNE)")
    ap.add_argument("--checkpoint", default="output_pretrain_zscore_2/checkpoint-tcga_luad_v2-500.pth",
                    help="체크포인트 경로")
    ap.add_argument("--pkl", default="datasets/rna_processed_zscore.pkl", help="rna_processed_zscore.pkl 경로")
    ap.add_argument("--out", default="output_pretrain_zscore_2/alignment_eval.png", help="t-SNE PNG 저장 경로")
    ap.add_argument("--max_patches", type=int, default=300)
    ap.add_argument("--n_genes", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    img_cls, omics_cls, n_samples = load_data_and_embeddings(
        args.checkpoint, args.pkl, str(device), max_patches=args.max_patches, n_genes=args.n_genes
    )
    print(f"임베딩 수: img_cls {img_cls.shape}, omics_cls {omics_cls.shape}")

    # 평가 1: Retrieval
    acc = retrieval_accuracy(img_cls, omics_cls, k_list=(1, 5, 10))
    print("\n=== Retrieval Accuracy ===")
    for name, v in acc.items():
        print(f"  {name}: {v:.2f}%")
    # Retrieval 결과 저장 (PNG와 같은 디렉터리, _retrieval.txt)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    retrieval_path = os.path.join(out_dir, os.path.splitext(os.path.basename(args.out))[0] + "_retrieval.txt")
    with open(retrieval_path, "w") as f:
        f.write(f"# Pre-training alignment retrieval (checkpoint: {args.checkpoint}, n_samples: {n_samples})\n")
        f.write("# === Retrieval Accuracy ===\n")
        for name, v in acc.items():
            f.write(f"{name}: {v:.2f}%\n")
    print(f"Retrieval 결과 저장: {retrieval_path}")

    # 평가 2: t-SNE
    tsne_visualization(img_cls, omics_cls, args.out)


if __name__ == "__main__":
    main()
