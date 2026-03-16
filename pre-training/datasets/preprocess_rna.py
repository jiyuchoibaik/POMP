"""
Step 1: RNA-seq 전처리
  STAR Counts TSV → HVG 선택 → z-score 정규화 (전체 환자 평균/표준편차 기준) → pickle 저장
  ※ pkl에 hvg_genes_mean, hvg_genes_std 를 함께 저장하여 fine-tuning 시 동일 정규화 적용 가능

사용법 (pre-training 폴더에서 실행 시, 데이터가 datasets/downloads/ 에 있을 때):
  python datasets/preprocess_rna.py \
    --rna_dir   ./datasets/downloads/rnaseq \
    --mapping   ./datasets/downloads/mapping.csv \
    --out       ./datasets/rna_processed_zscore.pkl \
    --n_genes   2000
  (z-score 버전은 rna_processed_zscore.pkl 등 다른 이름 권장, 기존 log1p pkl 유지)
"""

import os
import csv
import math
import pickle
import argparse
import glob
import numpy as np
from collections import defaultdict


def read_star_counts(tsv_path: str) -> dict:
    """
    STAR augmented_star_gene_counts.tsv 파싱
    컬럼: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
    unstranded 컬럼(raw counts) 사용
    """
    counts = {}
    with open(tsv_path) as f:
        for line in f:
            if line.startswith("N_") or line.startswith("gene_id"):
                continue  # 헤더/요약 행 스킵
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            gene_id = parts[0]   # ENSG...
            try:
                count = float(parts[3])  # unstranded raw count
            except ValueError:
                continue
            counts[gene_id] = count
    return counts


def select_hvg(all_counts: dict, n_genes: int = 2000) -> list:
    """
    분산이 높은 유전자(HVG) 선택
    all_counts: {case_id: {gene_id: count, ...}}
    """
    # 전체 케이스에 걸쳐 각 유전자의 분산 계산
    gene_vals = defaultdict(list)
    for counts in all_counts.values():
        for gid, val in counts.items():
            gene_vals[gid].append(val)

    # 모든 케이스에 있는 유전자만
    n_cases = len(all_counts)
    gene_vals = {g: v for g, v in gene_vals.items() if len(v) == n_cases}

    # log1p 후 분산 계산
    def variance(vals):
        log_vals = [math.log1p(v) for v in vals]
        mean = sum(log_vals) / len(log_vals)
        return sum((v - mean) ** 2 for v in log_vals) / len(log_vals)

    print(f"[HVG] 전체 공통 유전자: {len(gene_vals)}개 → 상위 {n_genes}개 선택")
    scored = sorted(gene_vals.keys(), key=lambda g: variance(gene_vals[g]), reverse=True)
    return scored[:n_genes]


# def log1p_normalize(counts: dict, hvg_genes: list) -> list:
#     """[변경 전] 선택된 HVG에 대해 log1p 정규화 후 리스트 반환"""
#     return [math.log1p(counts.get(g, 0.0)) for g in hvg_genes]


def zscore_normalize(matrix, axis=0, eps=1e-8):
    """
    전체 환자(axis=0) 기준 평균, 표준편차로 z-score 정규화.
    axis=0: 각 유전자(열)에 대해 환자들(행) 방향으로 평균/표준편차 계산.
    """
    mean = matrix.mean(axis=axis, keepdims=True)
    std = matrix.std(axis=axis, keepdims=True)
    std = np.where(std < eps, 1.0, std)  # std 0이면 1로 대체 (div by zero 방지)
    return (matrix - mean) / std, mean.squeeze(axis).tolist(), std.squeeze(axis).tolist()


def main(args):
    # mapping.csv 읽기 (없으면 경로 안내 후 종료)
    mapping_path = os.path.abspath(args.mapping)
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            f"mapping 파일 없음: {args.mapping}\n"
            f"  절대경로: {mapping_path}\n"
            f"  download_multimodal.py 로 다운로드 후 mapping.csv가 생성됩니다. "
            f"--mapping 에 실제 파일 경로를 지정하세요."
        )
    mapping = []
    with open(args.mapping) as f:
        mapping = list(csv.DictReader(f))

    paired = [r for r in mapping if r["paired"] == "True"]
    print(f"[INFO] 페어링된 케이스: {len(paired)}개")

    # TSV 파일 경로 찾기
    case_counts = {}
    missing = []
    for row in paired:
        cid    = row["case_id"]
        fname  = row["rna_file_name"]
        tsv    = os.path.join(args.rna_dir, cid, fname)

        if not os.path.exists(tsv):
            # 폴더 안에서 패턴으로 찾기
            found = glob.glob(os.path.join(args.rna_dir, cid, "*.tsv"))
            if found:
                tsv = found[0]
            else:
                missing.append(cid)
                continue

        case_counts[cid] = read_star_counts(tsv)

    if missing:
        print(f"[WARN] TSV 파일 없는 케이스 {len(missing)}개: {missing[:5]}")

    print(f"[INFO] TSV 로드 완료: {len(case_counts)}개")

    # HVG 선택
    hvg_genes = select_hvg(case_counts, n_genes=args.n_genes)

    # [변경 전] log1p 정규화
    # rna_vectors = {}
    # for cid, counts in case_counts.items():
    #     rna_vectors[cid] = log1p_normalize(counts, hvg_genes)

    # z-score 정규화: 먼저 log1p 스케일로 (N, n_genes) 행렬 구성 후, 전체 환자 기준 평균/표준편차로 z-score
    case_ids_ordered = list(case_counts.keys())
    n_cases = len(case_ids_ordered)
    log1p_matrix = np.zeros((n_cases, len(hvg_genes)), dtype=np.float64)
    for i, cid in enumerate(case_ids_ordered):
        counts = case_counts[cid]
        for j, g in enumerate(hvg_genes):
            log1p_matrix[i, j] = math.log1p(counts.get(g, 0.0))
    x_rna_z, hvg_genes_mean, hvg_genes_std = zscore_normalize(log1p_matrix, axis=0)
    rna_vectors = {cid: x_rna_z[i].tolist() for i, cid in enumerate(case_ids_ordered)}

    # WSI 경로 매핑
    wsi_paths = {}
    for row in paired:
        cid = row["case_id"]
        if cid in rna_vectors:
            wsi_paths[cid] = os.path.join(
                args.wsi_dir, cid, "regions.npy"
            ) if args.wsi_dir else ""

    # 저장 (fine-tuning 시 동일 z-score 적용을 위해 전체 환자 기준 평균/표준편차 포함)
    result = {
        "case_ids":        list(rna_vectors.keys()),
        "x_rna":           [rna_vectors[c] for c in rna_vectors],
        "wsi_paths":       [wsi_paths.get(c, "") for c in rna_vectors],
        "hvg_genes":       hvg_genes,
        "n_genes":         args.n_genes,
        "hvg_genes_mean":  hvg_genes_mean,   # 유전자별 전체 환자 평균 (z-score용)
        "hvg_genes_std":  hvg_genes_std,    # 유전자별 전체 환자 표준편차 (z-score용)
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    print(f"[DONE] 저장: {args.out}")
    print(f"  케이스 수  : {len(result['case_ids'])}")
    print(f"  RNA 차원   : {args.n_genes} (z-score 정규화)")
    print(f"  hvg_genes_mean/std 저장됨 → fine-tuning 시 동일 정규화 적용 가능")
    print(f"  예시 벡터  : {result['x_rna'][0][:5]} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_dir",  required=True, help="rnaseq TSV 디렉터리 (예: ./datasets/downloads/rnaseq)")
    ap.add_argument("--mapping",  required=True, help="mapping.csv 경로 (예: ./datasets/downloads/mapping.csv)")
    ap.add_argument("--wsi_dir",  default="", help="WSI regions.npy 루트 (미지정 시 wsi_paths는 빈 문자열)")
    ap.add_argument("--out",      default="./datasets/rna_processed_zscore.pkl",
                    help="z-score 버전은 별도 파일명 권장 (기존 rna_processed.pkl 덮어쓰지 않음)")
    ap.add_argument("--patch_dir", default="", help="extract_patches.py 출력 디렉토리")
    ap.add_argument("--n_genes",  default=2000, type=int, help="HVG 수 (default: 2000)")
    args = ap.parse_args()
    main(args)