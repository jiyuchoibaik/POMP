#!/usr/bin/env python3
"""
Survival 모델이 WSI(병리 영역)의 어느 패치에 주목하는지 cross-attention 기반으로 시각화.

- path_guided_omics_encoder의 cross-attention: Q=omics, K=V=image(WSI)
- 패치 그리드 + 히트맵 또는, coords 있으면 WSI 원본 썸네일 위에 attention 오버레이.

WSI 원본 위 시각화를 쓰려면:
  1) extract_patches.py로 패치 추출 시 coords.npz가 생성됨 (신규 추출)
  2) 이미 추출된 경우: extract_patches.py --coords_only ... 로 coords.npz만 생성
  3) 본 스크립트에 --wsi_dir ./pre-training/datasets/downloads/wsi 지정
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# survival 패키지 import를 위해 프로젝트 루트 추가
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from survival.utils.data_loader import POMPDataset


def _get_args():
    p = argparse.ArgumentParser(description="Survival WSI attention map visualization")
    p.add_argument("--pkl", required=True, help="CV splits pkl (e.g. luad_cv_splits.pkl)")
    p.add_argument("--checkpoint", required=True, help="Survival best model (e.g. best_model_tcga_luad_fold0.pth)")
    p.add_argument("--out", default="./output_finetune/attention_maps", help="Output directory for figures/csv")
    p.add_argument("--fold", type=int, default=0, help="Fold index to take test split from")
    p.add_argument("--sample_idx", type=int, default=0, help="Sample index within the chosen split")
    p.add_argument("--split", default="test", choices=("train", "validation", "test"),
                   help="Which split to take the sample from")
    p.add_argument("--max_num_region", type=int, default=250, help="Max regions per sample (must match training)")
    p.add_argument("--n_genes", type=int, default=2000, help="RNA dimension (must match training)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch_size", type=int, default=48,
                   help="Display patch size in grid (smaller = smaller figure)")
    p.add_argument("--grid_cols", type=int, default=None,
                   help="Grid columns for patch layout (default: auto from N_patches)")
    p.add_argument("--wsi_dir", default="",
                   help="WSI 루트 (예: ./pre-training/datasets/downloads/wsi). 주면 coords.npz 있을 때 원본 슬라이드 위 attention 오버레이 추가")
    p.add_argument("--thumb_size", type=int, default=1200,
                   help="WSI 썸네일 긴 변 최대 픽셀 (오버레이용)")
    p.add_argument("--top_pct", type=float, default=20.0,
                   help="상위 N%% attention만 강조 (해석용). 0이면 전체 표시, 20이면 상위 20%%만 색상/오버레이 (기본 20)")
    return p.parse_args()


def load_model(checkpoint_path: str, n_genes: int, device: torch.device):
    """Survival ViT 모델 생성 후 체크포인트 로드."""
    sys.path.insert(0, os.path.join(_REPO_ROOT, "survival"))
    from model import models_pomp

    model = models_pomp.vit_base_patch16(
        num_classes=1,
        drop_path_rate=0.1,
        global_pool=True,
        n_genes=n_genes,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def extract_attention(model, regions: torch.Tensor, X_rna: torch.Tensor, device: torch.device):
    """
    한 샘플에 대해 (regions, X_rna) forward 후 cross-attention 가중치 반환.

    Returns:
        attn_patches: (N_patch,) numpy, 각 WSI 패치에 대한 attention (CLS 제외)
    """
    regions = regions.unsqueeze(0).to(device)
    X_rna = X_rna.unsqueeze(0).to(device)
    samples = [regions, X_rna]

    with torch.no_grad():
        corr, image_embed, omics_embed = model(samples)
        _ = model.path_guided_omics_encoder(image_embed, omics_embed)

    # cross_attn[0].last_attn: (1, num_queries, num_keys) = (1, 2, N_patch+1)
    attn = model.cross_attn[0].last_attn
    attn = attn.cpu().float().numpy().squeeze(0)  # (2, N_patch+1)
    # 쿼리(omics) 축 평균 후, 이미지 CLS(index 0) 제거 → 패치만
    attn_patches = np.mean(attn, axis=0)[1:]  # (N_patch,)
    return attn_patches


def build_patch_grid(regions: np.ndarray, patch_size: int, grid_cols: int | None):
    """
    regions: (N, 3, 256, 256) or (N, C, H, W). (N,H,W,C) 형태면 (N,C,H,W)로 변환.
    패치들을 grid_cols 열로 배치한 한 장의 이미지로 만듦.
    """
    if regions.ndim == 4 and regions.shape[-1] in (1, 3):
        regions = np.transpose(regions, (0, 3, 1, 2)).copy()
    n = regions.shape[0]
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))

    import torch.nn.functional as F

    if isinstance(regions, np.ndarray):
        regions = torch.from_numpy(regions).float()
    if regions.dim() == 3:
        regions = regions.unsqueeze(0)
    # (N, C, H, W) -> N개 (C,H,W) 리사이즈
    small = F.interpolate(
        regions,
        size=(patch_size, patch_size),
        mode="bilinear",
        align_corners=False,
    )
    small = small.cpu().numpy()
    # (N, C, h, w) -> (N, h, w, C), 0~1로 정규화
    small = np.transpose(small, (0, 2, 3, 1))
    small = np.clip(small, 0, 1)
    if small.shape[-1] == 3:
        pass
    else:
        small = small[:, :, :, :3] if small.shape[-1] >= 3 else small

    # 빈 그리드
    h, w = patch_size, patch_size
    canvas = np.ones((grid_rows * h, grid_cols * w, 3), dtype=np.float32) * 0.9
    for i in range(n):
        row, col = i // grid_cols, i % grid_cols
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = small[i]
    return canvas, grid_rows, grid_cols


def plot_attention_map(
    regions: np.ndarray,
    attn_patches: np.ndarray,
    out_path: str,
    patch_size: int = 48,
    grid_cols: int | None = None,
    top_pct: float = 0.0,
):
    """실제 패치 그리드 + 동일 레이아웃의 attention 히트맵. top_pct>0이면 상위 N%만 색상으로 강조."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas, grid_rows, grid_cols = build_patch_grid(regions, patch_size, grid_cols)
    n = len(attn_patches)
    attn_2d = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    for i in range(n):
        r, c = i // grid_cols, i % grid_cols
        attn_2d[r, c] = attn_patches[i]

    # 상위 attention만 강조: threshold 이하는 회색으로 (해석용)
    if top_pct > 0 and top_pct < 100 and np.any(~np.isnan(attn_2d)):
        valid = attn_2d[~np.isnan(attn_2d)]
        if len(valid) > 0:
            thresh = np.percentile(valid, 100.0 - top_pct)
            attn_display = np.where(attn_2d >= thresh, attn_2d, np.nan)
        else:
            attn_display = attn_2d
    else:
        attn_display = attn_2d

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(canvas)
    axes[0].set_title("WSI regions (patch grid)")
    axes[0].axis("off")

    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad(color="lightgray", alpha=0.8)
    vmin = np.nanmin(attn_display) if np.any(~np.isnan(attn_display)) else 0
    vmax = np.nanmax(attn_display) if np.any(~np.isnan(attn_display)) else 1
    im = axes[1].imshow(attn_display, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    title = "Cross-attention (omics → WSI)"
    if top_pct > 0 and top_pct < 100:
        title += f" (top {int(top_pct)}% shown)"
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.6, label="Attention weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_wsi_original(wsi_path: str, out_path: str, thumb_size: int = 1200):
    """
    WSI 원본 썸네일만 저장 (어텐션/오버레이 없음).
    """
    import glob
    try:
        import openslide
    except ImportError:
        print("WSI 원본 저장을 위해 pip install openslide-python 필요.")
        return
    if os.path.isfile(wsi_path):
        svs_path = wsi_path
    else:
        cand = glob.glob(os.path.join(wsi_path, "*.svs"))
        if not cand:
            print(f"SVS 없음: {wsi_path}")
            return
        svs_path = cand[0]
    slide = openslide.OpenSlide(svs_path)
    level0_w, level0_h = slide.dimensions
    max_dim = max(level0_w, level0_h)
    if max_dim <= thumb_size:
        thumb_level = 0
    else:
        thumb_level = slide.level_count - 1
        for lv in range(slide.level_count):
            w, h = slide.level_dimensions[lv]
            if max(w, h) <= thumb_size:
                thumb_level = lv
                break
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumb = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h))
    thumb = np.array(thumb.convert("RGB"))
    slide.close()
    from PIL import Image
    Image.fromarray(thumb).save(out_path)
    print(f"Saved (WSI 원본): {out_path}")


def plot_wsi_attention_overlay(
    wsi_path: str,
    coords_npz_path: str,
    attn_patches: np.ndarray,
    out_path: str,
    thumb_size: int = 1200,
    top_pct: float = 0.0,
    regions: np.ndarray | None = None,
    patch_spatial_attn: np.ndarray | None = None,
):
    """
    WSI 썸네일 위에 패치별 attention 색상 + 최고 점수 패치 초록 박스.
    참고 레이아웃: 상단 WSI(패치 색칠 + 초록 박스), 하단 왼쪽 최고 패치 확대, 하단 오른쪽 해당 패치 attention 오버레이.
    coords.npz: 'coords' (N,2) level0 픽셀, 'downsample', 'patch_size'
    regions가 주어지면 하단 2패널(패치 원본 / 오버레이) 추가.
    """
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches as mpatches
    from matplotlib.colors import Normalize

    try:
        import openslide
    except ImportError:
        print("WSI 오버레이를 쓰려면 pip install openslide-python 필요. 패치 그리드만 저장합니다.")
        return

    if not os.path.isfile(coords_npz_path):
        print(f"coords 없음: {coords_npz_path} (extract_patches 또는 --coords_only 로 생성)")
        return

    # SVS 경로: wsi_dir/case_id/*.svs
    if os.path.isfile(wsi_path):
        svs_path = wsi_path
    else:
        cand = glob.glob(os.path.join(wsi_path, "*.svs"))
        if not cand:
            print(f"SVS 없음: {wsi_path}")
            return
        svs_path = cand[0]

    slide = openslide.OpenSlide(svs_path)
    npz = np.load(coords_npz_path, allow_pickle=True)
    coords = npz["coords"]  # (N, 2) level0
    ds_extract = float(npz["downsample"])
    patch_size_l0 = patch_size = int(npz.get("patch_size", 256))
    npz.close()

    n_patch = min(len(coords), len(attn_patches))
    coords = coords[:n_patch]
    attn = attn_patches[:n_patch].astype(np.float64)

    # 썸네일 레벨: 긴 변이 thumb_size 이하가 되도록
    level0_w, level0_h = slide.dimensions
    max_dim = max(level0_w, level0_h)
    if max_dim <= thumb_size:
        thumb_level = 0
    else:
        thumb_level = slide.level_count - 1
        for lv in range(slide.level_count):
            w, h = slide.level_dimensions[lv]
            if max(w, h) <= thumb_size:
                thumb_level = lv
                break
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumb_ds = float(slide.level_downsamples[thumb_level])

    # 썸네일 이미지 읽기
    thumb = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h))
    thumb = np.array(thumb.convert("RGB"))
    slide.close()

    # 패치 사각형을 썸네일 좌표로 (level0 -> thumb)
    patch_w_thumb = (patch_size_l0 * ds_extract) / thumb_ds
    patch_h_thumb = patch_w_thumb

    # 최고 attention 패치 인덱스
    best_idx = int(np.argmax(attn))
    attn_min, attn_max = attn.min(), attn.max()
    if attn_max <= attn_min:
        attn_max = attn_min + 1e-9
    norm = Normalize(vmin=attn_min, vmax=attn_max)
    cmap = plt.get_cmap("plasma")

    # 레이아웃: 상단 WSI, 하단 좌/우 (regions 있으면 3행 구성)
    if regions is not None and regions.shape[0] > 0 and best_idx < regions.shape[0]:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.25, wspace=0.2)
        ax_wsi = fig.add_subplot(gs[0, :])
        ax_patch = fig.add_subplot(gs[1, 0])
        ax_overlay = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_wsi = plt.subplots(1, 1, figsize=(8, 8))
        ax_patch = ax_overlay = None

    # 상단: WSI + 패치별 색상(일치도) + 최고 패치 초록 박스
    ax_wsi.imshow(thumb)
    for i in range(n_patch):
        x0 = coords[i, 0] / thumb_ds
        y0 = coords[i, 1] / thumb_ds
        color = cmap(norm(attn[i]))
        rect = mpatches.Rectangle((x0, y0), patch_w_thumb, patch_h_thumb,
                                  linewidth=0, facecolor=color, alpha=0.65)
        ax_wsi.add_patch(rect)
    # 최고 점수 패치 초록 테두리
    x0_best = coords[best_idx, 0] / thumb_ds
    y0_best = coords[best_idx, 1] / thumb_ds
    rect_best = mpatches.Rectangle((x0_best, y0_best), patch_w_thumb, patch_h_thumb,
                                   linewidth=3, edgecolor="lime", facecolor="none")
    ax_wsi.add_patch(rect_best)
    ax_wsi.axis("off")
    ax_wsi.set_title("WSI with patches colored by RNA-seq match (cross-attention); green box = highest score")
    # 컬러바: 패치 색상 = 일치도
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_wsi, shrink=0.5, aspect=20, label="Match score (attention)")

    # 하단 왼쪽: 최고 점수 패치 원본
    if ax_patch is not None and regions is not None:
        patch_img = regions[best_idx]
        if patch_img.ndim == 3 and patch_img.shape[0] in (1, 3):
            patch_img = np.transpose(patch_img, (1, 2, 0))
        if patch_img.shape[-1] == 1:
            patch_img = np.repeat(patch_img, 3, axis=-1)
        patch_img = np.clip(patch_img, 0, 1)
        ax_patch.imshow(patch_img)
        ax_patch.set_title("Highest matching patch")
        ax_patch.axis("off")

    # 하단 오른쪽: 해당 패치에 ViT self-attention 공간 맵 오버레이 (배경은 회색조로 해서 오버레이 대비)
    if ax_overlay is not None and regions is not None:
        patch_img = regions[best_idx]
        if patch_img.ndim == 3 and patch_img.shape[0] in (1, 3):
            patch_img = np.transpose(patch_img, (1, 2, 0)).copy()
        if patch_img.shape[-1] == 1:
            patch_img = np.repeat(patch_img, 3, axis=-1)
        patch_img = np.clip(patch_img, 0, 1)
        # 회색조로 표시해 노란/주황 attention이 잘 보이도록
        gray = np.dot(patch_img[..., :3], [0.299, 0.587, 0.114])
        ax_overlay.imshow(gray, cmap="gray")
        if patch_spatial_attn is not None and patch_spatial_attn.shape == (16, 16):
            from scipy.ndimage import zoom
            # (16, 16) → (256, 256) 업샘플
            attn_map = zoom(patch_spatial_attn, 16, order=1)
            attn_min, attn_max = attn_map.min(), attn_map.max()
            if attn_max > attn_min:
                attn_norm = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = np.ones_like(attn_map) * 0.5
            attn_norm = np.clip(attn_norm, 0, 1).astype(np.float32)
            cmap = plt.get_cmap("hot")
            overlay_rgb = cmap(attn_norm)[:, :, :3]
            overlay_rgba = np.dstack([overlay_rgb, 0.5 * attn_norm])
            ax_overlay.imshow(overlay_rgba)
        ax_overlay.set_title("Attention overlay on that patch")
        ax_overlay.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved (WSI overlay): {out_path}")


def main():
    args = _get_args()
    os.makedirs(args.out, exist_ok=True)

    # 데이터 로드
    import pickle
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
        print(f"sample_idx {args.sample_idx} >= len(dataset) {len(dataset)}, using 0")
        args.sample_idx = 0

    regions, X_rna, censored, survival = dataset[args.sample_idx]
    if isinstance(regions, np.ndarray):
        regions_np = regions
    else:
        regions_np = regions.numpy()

    # 모델 로드 및 attention 추출
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.n_genes, device)
    attn_patches = extract_attention(model, torch.from_numpy(regions_np).float(), X_rna, device)

    # CSV 저장 (나중에 WSI 좌표와 매핑할 때 유용)
    csv_path = os.path.join(args.out, f"attention_fold{args.fold}_sample{args.sample_idx}.csv")
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(len(attn_patches)), attn_patches]),
        delimiter=",",
        header="patch_index,attention",
        comments="",
    )
    print(f"Saved: {csv_path}")

    top_pct = getattr(args, "top_pct", 20.0)

    # 시각화: 패치 그리드 + 히트맵
    fig_path = os.path.join(args.out, f"attention_map_fold{args.fold}_sample{args.sample_idx}.png")
    plot_attention_map(
        regions_np,
        attn_patches,
        fig_path,
        patch_size=args.patch_size,
        grid_cols=args.grid_cols,
        top_pct=top_pct,
    )

    # WSI 원본 위 오버레이 (--wsi_dir 있고 coords.npz 있을 때)
    if getattr(args, "wsi_dir", "") and args.wsi_dir.strip():
        region_path = data[args.split]["region_pixel_5x"][args.sample_idx]
        case_dir = os.path.dirname(region_path)
        case_id = os.path.basename(case_dir)
        coords_npz = os.path.join(case_dir, "coords.npz")
        wsi_path = os.path.join(args.wsi_dir.strip(), case_id)
        # WSI 원본만 저장 (어텐션 오버레이 없음)
        out_original = os.path.join(args.out, f"wsi_original_fold{args.fold}_sample{args.sample_idx}.png")
        save_wsi_original(wsi_path, out_original, thumb_size=getattr(args, "thumb_size", 1200))
        out_wsi = os.path.join(args.out, f"attention_wsi_fold{args.fold}_sample{args.sample_idx}.png")
        # 최고 점수 패치에 대한 ViT 공간 attention (우측 하단 오버레이용)
        best_idx = int(np.argmax(attn_patches))
        patch_spatial_attn = None
        if hasattr(model, "get_patch_spatial_attention"):
            patch_tensor = torch.from_numpy(regions_np[best_idx : best_idx + 1]).float().to(device)
            patch_spatial_attn = model.get_patch_spatial_attention(patch_tensor)
        plot_wsi_attention_overlay(
            wsi_path,
            coords_npz,
            attn_patches,
            out_wsi,
            thumb_size=getattr(args, "thumb_size", 1200),
            top_pct=top_pct,
            regions=regions_np,
            patch_spatial_attn=patch_spatial_attn,
        )


if __name__ == "__main__":
    main()
