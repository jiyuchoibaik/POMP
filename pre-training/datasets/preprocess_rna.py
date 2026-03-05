"""
Step 1: RNA-seq 전처리
  STAR Counts TSV → HVG 선택 → log1p 정규화 → pickle 저장

사용법:
  python preprocess_rna.py \
    --rna_dir   ./downloads/rnaseq \
    --mapping   ./downloads/mapping.csv \
    --out       ./datasets/rna_processed.pkl \
    --n_genes   2000
"""

import os
import csv
import math
import pickle
import argparse
import glob
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


def log1p_normalize(counts: dict, hvg_genes: list) -> list:
    """
    선택된 HVG에 대해 log1p 정규화 후 리스트 반환
    """
    return [math.log1p(counts.get(g, 0.0)) for g in hvg_genes]


def main(args):
    # mapping.csv 읽기
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

    # 정규화
    rna_vectors = {}
    for cid, counts in case_counts.items():
        rna_vectors[cid] = log1p_normalize(counts, hvg_genes)

    # WSI 경로 매핑
    wsi_paths = {}
    for row in paired:
        cid = row["case_id"]
        if cid in rna_vectors:
            wsi_paths[cid] = os.path.join(
                args.wsi_dir, cid, "regions.npy"
            ) if args.wsi_dir else ""

    # 저장
    result = {
        "case_ids":  list(rna_vectors.keys()),
        "x_rna":     [rna_vectors[c] for c in rna_vectors],
        "wsi_paths": [wsi_paths.get(c, "") for c in rna_vectors],
        "hvg_genes": hvg_genes,
        "n_genes":   args.n_genes,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)

    print(f"[DONE] 저장: {args.out}")
    print(f"  케이스 수  : {len(result['case_ids'])}")
    print(f"  RNA 차원   : {args.n_genes}")
    print(f"  예시 벡터  : {result['x_rna'][0][:5]} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_dir",  required=True, help="downloads/rnaseq 경로")
    ap.add_argument("--mapping",  required=True, help="downloads/mapping.csv 경로")
    ap.add_argument("--out",      default="./datasets/rna_processed.pkl")
    ap.add_argument("--patch_dir", default="", help="extract_patches.py 출력 디렉토리")
    ap.add_argument("--n_genes",  default=2000, type=int, help="HVG 수 (default: 2000)")
    args = ap.parse_args()
    main(args)