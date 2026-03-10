"""
Step 3: luad_cv_splits.pkl 생성
  - RNA-seq pickle (preprocess_rna.py 출력)
  - 임상 데이터 CSV (download_clinical.py 출력)
  - WSI regions.npy 경로
  → survival/datasets/luad_cv_splits.pkl

사용법 (5-fold):
  python build_survival_pkl.py \
    --rna_pkl    ./datasets/rna_processed.pkl \
    --clinical   ./downloads/clinical.csv \
    --patch_dir  ./datasets/patches \
    --out        ./survival/datasets/luad_cv_splits.pkl \
    --n_folds    5 \
    --seed       42

단일 분할 (train/val/test 한 번만):
  python build_survival_pkl.py \
    --rna_pkl    ... --clinical ... --patch_dir ... \
    --out        ./survival/datasets/luad_single_splits.pkl \
    --n_folds    1 \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
    --seed       42
"""

import os
import csv
import pickle
import random
import argparse
import numpy as np
from collections import defaultdict


def load_rna(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # {case_id: rna_vector}
    return {cid: np.array(vec, dtype=np.float32)
            for cid, vec in zip(data["case_ids"], data["x_rna"])}


def load_clinical(csv_path: str):
    clinical = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = row["case_id"].upper()
            clinical[cid] = {
                "survival": float(row["os_days"]),
                "censored": int(row["censored"]),  # 1=사망, 0=중도절단
            }
    return clinical


def find_regions_npy(patch_dir: str, case_id: str) -> str:
    """patch_dir/{case_id}/regions.npy 경로 반환"""
    path = os.path.join(patch_dir, case_id, "regions.npy")
    return path if os.path.exists(path) else ""


def build_single_split(case_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    한 번만 나누기: train / val / test 비율로 1회 분할.
    Returns: [ {train: [...], validation: [...], test: [...]} ]
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    np.random.seed(seed)
    shuffled = case_ids.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val
    return [{
        "train":      shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test":       shuffled[n_train + n_val :],
    }]


def build_splits(case_ids, n_folds=5, seed=42):
    """
    Stratified k-fold: 사망/생존 비율 유지
    Returns: [{train: [...], validation: [...], test: [...]}, ...]
    """
    random.seed(seed)
    np.random.seed(seed)

    shuffled = case_ids.copy()
    random.shuffle(shuffled)

    folds = [[] for _ in range(n_folds)]
    for i, cid in enumerate(shuffled):
        folds[i % n_folds].append(cid)

    splits = []
    for k in range(n_folds):
        test_ids  = folds[k]
        val_ids   = folds[(k + 1) % n_folds]
        train_ids = []
        for j in range(n_folds):
            if j != k and j != (k + 1) % n_folds:
                train_ids.extend(folds[j])
        splits.append({
            "train":      train_ids,
            "validation": val_ids,
            "test":       test_ids,
        })
    return splits


def pack_split(case_ids, rna_map, clinical_map, patch_dir):
    x_rna         = []
    censored      = []
    survival      = []
    region_paths  = []

    for cid in case_ids:
        x_rna.append(rna_map[cid])
        censored.append(clinical_map[cid]["censored"])
        survival.append(clinical_map[cid]["survival"])
        region_paths.append(find_regions_npy(patch_dir, cid))

    return {
        "x_rna":          np.array(x_rna,        dtype=np.float32),   # (N, n_genes)
        "censored":        np.array(censored,      dtype=np.int64),     # (N,)
        "survival":        np.array(survival,      dtype=np.float32),   # (N,) days
        "region_pixel_5x": np.array(region_paths, dtype=object),       # (N,) str paths
    }


def main(args):
    print("[1] RNA-seq 로드...")
    rna_map = load_rna(args.rna_pkl)
    print(f"    {len(rna_map)}개, 차원={next(iter(rna_map.values())).shape[0]}")

    print("[2] 임상 데이터 로드...")
    clinical_map = load_clinical(args.clinical)
    print(f"    {len(clinical_map)}개")

    # 교집합: RNA + 임상 + regions.npy 모두 있는 케이스
    valid_ids = []
    no_clinical = []
    no_patch    = []

    for cid in rna_map:
        if cid not in clinical_map:
            no_clinical.append(cid)
            continue
        npy = find_regions_npy(args.patch_dir, cid)
        if not npy:
            no_patch.append(cid)
            continue
        valid_ids.append(cid)

    print(f"\n[3] 유효 케이스: {len(valid_ids)}")
    if no_clinical:
        print(f"    임상 없음: {len(no_clinical)}개")
    if no_patch:
        print(f"    regions.npy 없음: {len(no_patch)}개  "
              f"(extract_patches.py 실행 후 재시도)")

    if len(valid_ids) == 0:
        raise RuntimeError(
            "유효 케이스가 없습니다. "
            "extract_patches.py와 download_clinical.py를 먼저 실행하세요."
        )

    # 생존 통계
    dead = sum(1 for c in valid_ids if clinical_map[c]["censored"] == 1)
    print(f"    사망: {dead} / 중도절단: {len(valid_ids)-dead}")
    survivals = [clinical_map[c]["survival"] for c in valid_ids]
    print(f"    생존기간: min={min(survivals):.0f}d, "
          f"median={sorted(survivals)[len(survivals)//2]:.0f}d, "
          f"max={max(survivals):.0f}d")

    if args.n_folds == 1:
        print(f"\n[4] 단일 분할 (train/val/test = {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%})...")
        splits = build_single_split(
            valid_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        print(f"\n[4] {args.n_folds}-fold 분할 생성...")
        splits = build_splits(valid_ids, n_folds=args.n_folds, seed=args.seed)

    data_cv_splits = {}
    for k, split in enumerate(splits):
        data_cv_splits[k] = {
            "train":      pack_split(split["train"],      rna_map, clinical_map, args.patch_dir),
            "validation": pack_split(split["validation"], rna_map, clinical_map, args.patch_dir),
            "test":       pack_split(split["test"],       rna_map, clinical_map, args.patch_dir),
        }
        print(f"    fold {k}: train={len(split['train'])}, "
              f"val={len(split['validation'])}, "
              f"test={len(split['test'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(data_cv_splits, f)

    print(f"\n[DONE] 저장: {args.out}")
    print(f"  pkl 키: {list(data_cv_splits[0]['train'].keys())}")
    print(f"  x_rna shape: {data_cv_splits[0]['train']['x_rna'].shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_pkl",   required=True, help="preprocess_rna.py 출력 pkl")
    ap.add_argument("--clinical",  required=True, help="download_clinical.py 출력 csv")
    ap.add_argument("--patch_dir", required=True, help="extract_patches.py 출력 디렉토리")
    ap.add_argument("--out",       default="./survival/datasets/luad_cv_splits.pkl")
    ap.add_argument("--n_folds",   default=5, type=int,
                    help="1이면 단일 분할(train/val/test 비율), 2 이상이면 k-fold")
    ap.add_argument("--train_ratio", default=0.7, type=float, help="n_folds=1일 때 train 비율")
    ap.add_argument("--val_ratio",   default=0.15, type=float, help="n_folds=1일 때 validation 비율")
    ap.add_argument("--test_ratio", default=0.15, type=float, help="n_folds=1일 때 test 비율")
    ap.add_argument("--seed",      default=42, type=int)
    args = ap.parse_args()
    main(args)