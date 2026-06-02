"""
WSI-CLS와 RNA-CLS 간 코사인 유사도 분석 스크립트
사전 학습된 체크포인트를 로드하여 두 모달리티의 표현이
실질적으로 얼마나 가까운지 확인합니다.

사용법:
  python check_alignment.py \
    --checkpoint ./output_pretrain_zscore_2/checkpoint-400.pth \
    --data_pkl   ./datasets/rna_processed_zscore.pkl \
    --max_patches 300
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.models_pomp import vit_base_patch16
from utils.data_loader import build_dataset


def load_model(checkpoint_path: str, n_genes: int = 2000, device: torch.device = None):
    model = vit_base_patch16(
        rna_dim        = n_genes,
        drop_path_rate = 0.1,
        global_pool    = True,
        num_classes    = 0,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_similarities(model, data_loader, device):
    wsi_cls_list  = []
    rna_cls_list  = []

    for regions, x_rna in data_loader:
        regions = regions.to(device)
        x_rna   = x_rna.to(device)

        img_cls, omics_cls, _, _ = model([regions, x_rna])

        # (B, 1, D) → (B, D)
        wsi_cls_list.append(img_cls.squeeze(1).cpu())
        rna_cls_list.append(omics_cls.squeeze(1).cpu())

    wsi_cls  = torch.cat(wsi_cls_list,  dim=0)  # (N, D)
    rna_cls  = torch.cat(rna_cls_list,  dim=0)  # (N, D)
    return wsi_cls, rna_cls


def analyze(wsi_cls, rna_cls):
    # ── 1. 대응 쌍 코사인 유사도 ──────────────────────────────────────
    cos_sim = F.cosine_similarity(wsi_cls, rna_cls, dim=1)  # (N,)

    print("=" * 50)
    print("[ 대응 쌍 코사인 유사도 ]")
    print(f"  평균  : {cos_sim.mean().item():.4f}")
    print(f"  표준편차: {cos_sim.std().item():.4f}")
    print(f"  최솟값: {cos_sim.min().item():.4f}")
    print(f"  최댓값: {cos_sim.max().item():.4f}")
    print()
    print("  ※ 해석 기준")
    print("    0.95 이상 → 두 표현이 거의 동일 (mode collapse 의심)")
    print("    0.70~0.95 → 정렬됐으나 각자 특성 유지 (이상적)")
    print("    0.70 미만  → 정렬 부족")
    print()

    # ── 2. 비대응 쌍 코사인 유사도 (랜덤 100쌍) ───────────────────────
    N = wsi_cls.shape[0]
    idx = torch.randperm(N)[:min(100, N)]
    # 비대응: wsi[i] vs rna[i+1 mod N]
    neg_rna = rna_cls[torch.roll(idx, 1)]
    cos_neg = F.cosine_similarity(wsi_cls[idx], neg_rna, dim=1)

    print("[ 비대응 쌍 코사인 유사도 (랜덤 100쌍) ]")
    print(f"  평균  : {cos_neg.mean().item():.4f}")
    print(f"  표준편차: {cos_neg.std().item():.4f}")
    print()
    print("  ※ 대응 쌍 유사도와의 차이가 클수록 정렬이 잘 된 것")
    print()

    # ── 3. L2 거리 ────────────────────────────────────────────────────
    l2_dist = (wsi_cls - rna_cls).norm(dim=1)
    print("[ 대응 쌍 L2 거리 ]")
    print(f"  평균  : {l2_dist.mean().item():.4f}")
    print(f"  표준편차: {l2_dist.std().item():.4f}")
    print()

    # ── 4. 벡터 차이의 크기 비율 ─────────────────────────────────────
    wsi_norm = wsi_cls.norm(dim=1)
    rna_norm = rna_cls.norm(dim=1)
    diff_ratio = l2_dist / ((wsi_norm + rna_norm) / 2)
    print("[ L2 거리 / 평균 노름 비율 ]")
    print(f"  평균  : {diff_ratio.mean().item():.4f}")
    print(f"  ※ 0에 가까울수록 두 벡터가 유사, 1 이상이면 실질적으로 다름")
    print("=" * 50)

    # ── 5. 요약 판정 ──────────────────────────────────────────────────
    mean_cos = cos_sim.mean().item()
    print()
    print("[ 최종 판정 ]")
    if mean_cos >= 0.95:
        print("  ⚠️  평균 코사인 유사도가 0.95 이상입니다.")
        print("      두 모달리티 표현이 거의 동일할 가능성이 있습니다.")
        print("      대조 학습이 과도하게 작동했을 수 있습니다.")
    elif mean_cos >= 0.70:
        print("  ✅  평균 코사인 유사도가 0.70~0.95 범위입니다.")
        print("      정렬은 이루어졌으나 각 모달리티 특성이 유지되고 있습니다.")
    else:
        print("  ❌  평균 코사인 유사도가 0.70 미만입니다.")
        print("      정렬이 충분히 이루어지지 않았을 수 있습니다.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print("[INFO] 모델 로드 중...")
    model = load_model(args.checkpoint, n_genes=args.n_genes, device=device)

    print("[INFO] 데이터 로드 중...")
    dataset = build_dataset(args.data_pkl, max_num_region=args.max_patches)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = 4,
        shuffle     = False,
        drop_last   = False,
    )
    print(f"[INFO] 총 {len(dataset)}개 케이스")

    print("[INFO] 표현 추출 중...")
    wsi_cls, rna_cls = compute_similarities(model, data_loader, device)

    analyze(wsi_cls, rna_cls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True, help="사전 학습 체크포인트 경로")
    ap.add_argument("--data_pkl",    default="./datasets/rna_processed_zscore.pkl")
    ap.add_argument("--n_genes",     default=2000,  type=int)
    ap.add_argument("--max_patches", default=300,   type=int)
    ap.add_argument("--batch_size",  default=4,     type=int)
    args = ap.parse_args()
    main(args)