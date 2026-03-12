#!/usr/bin/env python3
"""
멀티모달 vs 유니모달 WSI 집중 영역 비교.

- 멀티모달: pre-training (POC+MOM+POM) 체크포인트 → 이미지 인코더가 오믹스와 정렬된 뒤의 CLS→region attention.
- 유니모달: 이미지 단일 SSL 체크포인트 → 오믹스 없이 학습한 이미지 인코더의 CLS→region attention.

동일 WSI 샘플에 대해 두 attention을 나란히 시각화 (패치 그리드 + 가능하면 WSI 오버레이).

사용 예:
  cd /path/to/POMP
  python survival/scripts/compare_multimodal_unimodal_attention.py \\
    --pkl ./survival/datasets/luad_cv_splits.pkl \\
    --multimodal_ckpt ./pre-training/output_finetune_dir/checkpoint-poc_mom-100.pth \\
    --unimodal_ckpt ./pre-training/output_unimodal_ssl/checkpoint-unimodal-last.pth \\
    --out ./output_finetune/attention_maps \\
    --wsi_dir ./pre-training/datasets/downloads/wsi \\
    --fold 0 --sample_idx 0
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# pre-training 모델 사용
_PRETRAIN_ROOT = os.path.join(_REPO_ROOT, "pre-training")
if _PRETRAIN_ROOT not in sys.path:
    sys.path.insert(0, _PRETRAIN_ROOT)

from survival.utils.data_loader import POMPDataset


def _get_args():
    p = argparse.ArgumentParser(description="Multimodal vs Unimodal WSI attention comparison")
    p.add_argument("--pkl", required=True, help="CV splits pkl (luad_cv_splits.pkl)")
    p.add_argument("--multimodal_ckpt", required=True, help="Pre-training 체크포인트 (POC+MOM+POM)")
    p.add_argument("--unimodal_ckpt", required=True, help="Unimodal SSL 체크포인트 (main_unimodal_ssl.py 출력)")
    p.add_argument("--out", default="./output_finetune/attention_maps")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--split", default="test", choices=("train", "validation", "test"))
    p.add_argument("--max_num_region", type=int, default=250)
    p.add_argument("--wsi_dir", default="", help="WSI 루트. 있으면 coords.npz 있을 때 WSI 오버레이 추가")
    p.add_argument("--thumb_size", type=int, default=1200)
    p.add_argument("--top_pct", type=float, default=20.0, help="상위 N%%만 강조. 0이면 전체")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_pretraining_model(ckpt_path: str, device: torch.device):
    from model.models_pomp import vit_base_patch16
    model = vit_base_patch16(rna_dim=2000)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def plot_side_by_side(
    regions: np.ndarray,
    attn_multi: np.ndarray,
    attn_uni: np.ndarray,
    out_path: str,
    top_pct: float = 20.0,
):
    """패치 그리드 + 두 attention 히트맵 나란히 (멀티 | 유니모달)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(attn_multi)
    grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))

    def to_2d(a):
        a2d = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
        for i in range(len(a)):
            r, c = i // grid_cols, i % grid_cols
            a2d[r, c] = a[i]
        return a2d

    if top_pct > 0 and top_pct < 100:
        for attn in (attn_multi, attn_uni):
            valid = attn[~np.isnan(attn)]
            if len(valid) > 0:
                thresh = np.percentile(valid, 100.0 - top_pct)
                attn[attn < thresh] = np.nan
    attn_multi_2d = to_2d(attn_multi)
    attn_uni_2d = to_2d(attn_uni)

    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad(color="lightgray", alpha=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin = min(np.nanmin(attn_multi_2d), np.nanmin(attn_uni_2d))
    vmax = max(np.nanmax(attn_multi_2d), np.nanmax(attn_uni_2d))
    if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    axes[0].imshow(attn_multi_2d, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    axes[0].set_title("Multimodal (POC+MOM+POM)\nimage encoder CLS→region attention")
    axes[0].axis("off")
    axes[1].imshow(attn_uni_2d, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    axes[1].set_title("Unimodal (image-only SSL)\nimage encoder CLS→region attention")
    axes[1].axis("off")
    if top_pct > 0 and top_pct < 100:
        fig.suptitle(f"Top {int(top_pct)}% attention only (interpretability)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_wsi_overlay_pair(
    wsi_path: str,
    coords_npz_path: str,
    attn_multi: np.ndarray,
    attn_uni: np.ndarray,
    out_path: str,
    thumb_size: int = 1200,
    top_pct: float = 20.0,
):
    """WSI 썸네일 위에 멀티/유니모달 attention 나란히 오버레이."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    try:
        import openslide
    except ImportError:
        print("openslide 없음. WSI 오버레이 스킵.")
        return
    if not os.path.isfile(coords_npz_path):
        print(f"coords 없음: {coords_npz_path}")
        return
    if os.path.isfile(wsi_path):
        svs_path = wsi_path
    else:
        cand = glob.glob(os.path.join(wsi_path, "*.svs"))
        if not cand:
            return
        svs_path = cand[0]

    slide = openslide.OpenSlide(svs_path)
    npz = np.load(coords_npz_path, allow_pickle=True)
    coords = npz["coords"]
    ds_extract = float(npz["downsample"])
    patch_size = int(npz.get("patch_size", 256))
    npz.close()

    n_patch = min(len(coords), len(attn_multi), len(attn_uni))
    coords = coords[:n_patch]
    cx = coords[:, 0] + patch_size * ds_extract / 2.0
    cy = coords[:, 1] + patch_size * ds_extract / 2.0

    thumb_level = slide.level_count - 1
    for lv in range(slide.level_count):
        w, h = slide.level_dimensions[lv]
        if max(w, h) <= thumb_size:
            thumb_level = lv
            break
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumb_ds = slide.level_downsamples[thumb_level]
    thumb = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h))
    thumb = np.array(thumb.convert("RGB"))
    slide.close()

    tx = np.clip((cx / thumb_ds).astype(int), 0, thumb_w - 1)
    ty = np.clip((cy / thumb_ds).astype(int), 0, thumb_h - 1)
    sigma = max(thumb_w, thumb_h) / 120.0

    def make_heat(attn, mask_top):
        heat = np.zeros((thumb_h, thumb_w), dtype=np.float64)
        for i in range(n_patch):
            if mask_top[i]:
                heat[ty[i], tx[i]] = attn[i]
        heat = gaussian_filter(heat, sigma=sigma, mode="constant", cval=0)
        h_min, h_max = heat.min(), heat.max()
        if h_max <= h_min:
            h_max = h_min + 1e-9
        return (heat - h_min) / (h_max - h_min)

    if top_pct > 0 and top_pct < 100:
        thresh_m = np.percentile(attn_multi[:n_patch], 100.0 - top_pct)
        thresh_u = np.percentile(attn_uni[:n_patch], 100.0 - top_pct)
        mask_m = attn_multi[:n_patch] >= thresh_m
        mask_u = attn_uni[:n_patch] >= thresh_u
    else:
        mask_m = np.ones(n_patch, dtype=bool)
        mask_u = np.ones(n_patch, dtype=bool)

    heat_m = make_heat(attn_multi[:n_patch], mask_m)
    heat_u = make_heat(attn_uni[:n_patch], mask_u)
    alpha = 0.65

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(thumb)
    axes[0].imshow(heat_m, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    axes[0].set_title("Multimodal (aligned with omics)")
    axes[0].axis("off")
    axes[1].imshow(thumb)
    axes[1].imshow(heat_u, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title("Unimodal (image-only SSL)")
    axes[1].axis("off")
    if top_pct > 0 and top_pct < 100:
        fig.suptitle(f"WSI attention (top {int(top_pct)}%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (WSI pair): {out_path}")


def main():
    args = _get_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.pkl, "rb") as f:
        cv_splits = pickle.load(f)
    if isinstance(cv_splits, dict) and args.fold in cv_splits:
        data = cv_splits[args.fold]
    elif isinstance(cv_splits, list):
        data = cv_splits[args.fold]
    else:
        data = list(cv_splits.values())[args.fold]

    dataset = POMPDataset(data=data, split=args.split, max_num_region=args.max_num_region)
    if args.sample_idx >= len(dataset):
        args.sample_idx = 0
    regions, _, _, _ = dataset[args.sample_idx]
    if hasattr(regions, "numpy"):
        regions = regions.numpy()
    regions = np.asarray(regions)
    if regions.ndim == 4 and regions.shape[-1] in (1, 3):
        regions = np.transpose(regions, (0, 3, 1, 2))

    device = torch.device(args.device)
    model_multi = load_pretraining_model(args.multimodal_ckpt, device)
    model_uni = load_pretraining_model(args.unimodal_ckpt, device)

    regions_t = torch.from_numpy(regions).float().unsqueeze(0).to(device)
    with torch.no_grad():
        attn_multi = model_multi.get_image_cls_region_attention(regions_t).squeeze(0)
        attn_uni = model_uni.get_image_cls_region_attention(regions_t).squeeze(0)

    attn_multi = np.asarray(attn_multi)
    attn_uni = np.asarray(attn_uni)

    out_grid = os.path.join(args.out, f"compare_multi_uni_fold{args.fold}_sample{args.sample_idx}.png")
    plot_side_by_side(regions, attn_multi.copy(), attn_uni.copy(), out_grid, top_pct=args.top_pct)

    if args.wsi_dir.strip():
        region_path = data[args.split]["region_pixel_5x"][args.sample_idx]
        case_dir = os.path.dirname(region_path)
        coords_npz = os.path.join(case_dir, "coords.npz")
        case_id = os.path.basename(case_dir)
        wsi_path = os.path.join(args.wsi_dir.strip(), case_id)
        out_wsi = os.path.join(args.out, f"compare_multi_uni_wsi_fold{args.fold}_sample{args.sample_idx}.png")
        plot_wsi_overlay_pair(
            wsi_path, coords_npz, attn_multi, attn_uni, out_wsi,
            thumb_size=args.thumb_size, top_pct=args.top_pct,
        )


if __name__ == "__main__":
    main()
