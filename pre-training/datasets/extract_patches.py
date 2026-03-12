"""
Step 2: SVS → regions.npy 패치 추출
  openslide로 SVS를 읽어 256x256 패치를 추출하고 regions.npy로 저장

사용법:
  pip install openslide-python numpy Pillow tqdm

  python extract_patches.py \
    --wsi_dir  ./downloads/wsi \
    --out_dir  ./datasets/patches \
    --mapping  ./downloads/mapping.csv \
    --mag      5 \
    --patch_size 256 \
    --max_patches 300
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path

try:
    import openslide
except ImportError:
    raise ImportError("pip install openslide-python 먼저 실행하세요")

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow 먼저 실행하세요")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def get_best_level(slide, target_mag: int = 5) -> int:
    """목표 배율에 가장 가까운 레벨 반환"""
    try:
        scan_mag = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 20))
    except Exception:
        scan_mag = 20.0

    downsample_target = scan_mag / target_mag
    downsamples = slide.level_downsamples
    diffs = [abs(ds - downsample_target) for ds in downsamples]
    return int(np.argmin(diffs))


def is_tissue(patch_arr: np.ndarray, threshold: float = 0.7) -> bool:
    """
    배경(흰색) 비율로 조직 유무 판단
    patch_arr: (H, W, 3) uint8
    """
    gray  = 0.299*patch_arr[...,0] + 0.587*patch_arr[...,1] + 0.114*patch_arr[...,2]
    white = (gray > 220).mean()
    return white < threshold


def extract_patches(svs_path: str, out_npy: str,
                    mag: int = 5, patch_size: int = 256,
                    max_patches: int = 300) -> bool:
    if os.path.exists(out_npy):
        print(f"  [SKIP] {os.path.basename(out_npy)}")
        return True

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        print(f"  [ERROR] SVS 열기 실패: {e}")
        return False

    level    = get_best_level(slide, mag)
    lw, lh   = slide.level_dimensions[level]
    ds       = slide.level_downsamples[level]

    patches = []
    coords  = []  # level0 기준 (x0, y0), WSI 원본 위 attention 시각화용
    n_x = lw // patch_size
    n_y = lh // patch_size

    for yi in range(n_y):
        for xi in range(n_x):
            # level 좌표 → level0 좌표
            x0 = int(xi * patch_size * ds)
            y0 = int(yi * patch_size * ds)

            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            arr    = np.array(region.convert("RGB"))  # (256, 256, 3)

            if is_tissue(arr):
                # (H, W, C) → (C, H, W), float32 [0,1]
                patches.append(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)
                coords.append([x0, y0])

            if len(patches) >= max_patches:
                break
        if len(patches) >= max_patches:
            break

    slide.close()

    if len(patches) == 0:
        print(f"  [WARN] 조직 패치 없음: {svs_path}")
        return False

    patches_arr = np.stack(patches, axis=0)  # (N, 3, 256, 256)
    coords_arr  = np.array(coords, dtype=np.int64)   # (N, 2) level0 픽셀
    out_dir = os.path.dirname(out_npy)
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_npy, patches_arr)
    np.savez(os.path.join(out_dir, "coords.npz"), coords=coords_arr, downsample=float(ds), patch_size=patch_size)
    return True


def extract_coords_only(svs_path: str, out_coords_npy: str,
                        mag: int = 5, patch_size: int = 256,
                        max_patches: int = 300) -> bool:
    """
    이미 존재하는 regions.npy와 동일한 순서로 coords.npz만 생성.
    (과거에 좌표 없이 추출한 경우, WSI 원본 위 시각화를 위해 사용)
    """
    out_npz = out_coords_npy.replace(".npy", ".npz")
    if os.path.exists(out_npz):
        return True
    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception:
        return False

    level = get_best_level(slide, mag)
    lw, lh = slide.level_dimensions[level]
    ds = slide.level_downsamples[level]

    coords = []
    n_x = lw // patch_size
    n_y = lh // patch_size

    for yi in range(n_y):
        for xi in range(n_x):
            x0 = int(xi * patch_size * ds)
            y0 = int(yi * patch_size * ds)
            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            arr = np.array(region.convert("RGB"))
            if is_tissue(arr):
                coords.append([x0, y0])
            if len(coords) >= max_patches:
                break
        if len(coords) >= max_patches:
            break
    slide.close()

    if len(coords) == 0:
        return False
    coords_arr = np.array(coords, dtype=np.int64)
    out_dir = os.path.dirname(out_coords_npy)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, coords=coords_arr, downsample=float(ds), patch_size=patch_size)
    return True


def main(args):
    # mapping.csv 읽기
    with open(args.mapping) as f:
        mapping = [r for r in csv.DictReader(f) if r["paired"] == "True"]

    if getattr(args, "coords_only", False):
        print("[INFO] 기존 regions.npy에 맞춰 coords.npy만 생성")
        ok = fail = 0
        for row in tqdm(mapping, desc="coords 생성"):
            cid = row["case_id"]
            wsi_name = row["wsi_file_name"]
            svs_path = os.path.join(args.wsi_dir, cid, wsi_name)
            out_coords = os.path.join(args.out_dir, cid, "coords.npy")
            out_regions = os.path.join(args.out_dir, cid, "regions.npy")
            if not os.path.exists(out_regions):
                fail += 1
                continue
            if not os.path.exists(svs_path):
                fail += 1
                continue
            if extract_coords_only(svs_path, out_coords, args.mag, args.patch_size, args.max_patches):
                ok += 1
            else:
                fail += 1
        print(f"[DONE] coords.npy 성공: {ok} | 실패: {fail}")
        return

    print(f"[INFO] 패치 추출 대상: {len(mapping)}개")
    ok = fail = 0

    for row in tqdm(mapping, desc="패치 추출"):
        cid      = row["case_id"]
        wsi_name = row["wsi_file_name"]
        svs_path = os.path.join(args.wsi_dir, cid, wsi_name)
        out_npy  = os.path.join(args.out_dir, cid, "regions.npy")

        if not os.path.exists(svs_path):
            print(f"  [SKIP] SVS 없음: {svs_path}")
            fail += 1
            continue

        if extract_patches(svs_path, out_npy,
                           args.mag, args.patch_size, args.max_patches):
            ok += 1
        else:
            fail += 1

    print(f"\n[DONE] 성공: {ok} | 실패/스킵: {fail}")
    print(f"  저장 위치: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir",     required=True, help="downloads/wsi 경로")
    ap.add_argument("--out_dir",     default="./datasets/patches")
    ap.add_argument("--mapping",     required=True, help="downloads/mapping.csv")
    ap.add_argument("--mag",         default=5,   type=int, help="목표 배율 (default: 5x)")
    ap.add_argument("--patch_size",  default=256, type=int)
    ap.add_argument("--max_patches", default=300, type=int)
    ap.add_argument("--coords_only", action="store_true",
                    help="regions.npy는 이미 있을 때 coords.npy만 생성 (WSI 원본 시각화용)")
    args = ap.parse_args()
    main(args)