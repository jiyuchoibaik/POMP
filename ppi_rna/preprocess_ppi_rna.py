"""
PPI 기반 RNA 클러스터 전처리
  - 기존 rna_processed.pkl (HVG 2000) 로드
  - STRING DB로 PPI 네트워크 구축 후 Louvain 클러스터링
  - 클러스터별 평균 발현값으로 x_rna (N, n_clusters) 생성

사용법:
  python preprocess_ppi_rna.py \
    --rna_pkl ../pre-training/datasets/rna_processed.pkl \
    --rna_dir ../pre-training/datasets/../downloads/rnaseq \
    --mapping ../pre-training/datasets/../downloads/mapping.csv \
    --out ./datasets/rna_ppi_clusters.pkl \
    [--string_cache ./string_cache] [--min_ppi_score 150]
"""
import os
import sys
import csv
import glob
import pickle
import argparse
import numpy as np
from collections import defaultdict

# STRING + 그래프
try:
    import networkx as nx
except ImportError:
    print("pip install networkx")
    sys.exit(1)
try:
    import community as community_louvain
except ImportError:
    try:
        import louvain as community_louvain
    except ImportError:
        print("pip install python-louvain  # or pip install louvain")
        sys.exit(1)

from string_utils import (
    get_string_ids_for_genes,
    load_string_id_to_name_from_info,
    load_ppi_edges,
    download_string_file,
    HUMAN_TAX_ID,
)


def read_star_gene_names(tsv_path: str) -> dict:
    """STAR TSV에서 gene_id -> gene_name 매핑 (컬럼 0, 1)."""
    out = {}
    with open(tsv_path) as f:
        for line in f:
            if line.startswith("N_") or line.startswith("gene_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def get_ensg_to_gene_name(rna_dir: str, mapping_path: str, hvg_genes: list) -> dict:
    """mapping + rna_dir에서 TSV 하나 골라 해당 TSV에서 HVG의 ENSG -> gene_name 반환."""
    with open(mapping_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("paired") == "True"]
    for row in rows:
        cid = row["case_id"]
        fname = row.get("rna_file_name", "")
        tsv = os.path.join(rna_dir, cid, fname)
        if not os.path.isfile(tsv):
            found = glob.glob(os.path.join(rna_dir, cid, "*.tsv"))
            tsv = found[0] if found else ""
        if not tsv or not os.path.isfile(tsv):
            continue
        id_to_name = read_star_gene_names(tsv)
        return {g: id_to_name.get(g, g) for g in hvg_genes}
    return {g: g for g in hvg_genes}


def build_gene_index_to_cluster(
    hvg_genes: list,
    ensg_to_name: dict,
    name_to_string_id: dict,
    links_path: str,
    min_ppi_score: int,
) -> tuple:
    """
    PPI 그래프 구축 -> Louvain 클러스터링 -> gene_index -> cluster_id.
    Returns:
        gene_to_cluster: list of length len(hvg_genes), each element = cluster_id (int)
        n_clusters: int
    """
    string_ids = set()
    gene_idx_to_string_id = []
    for i, ensg in enumerate(hvg_genes):
        name = ensg_to_name.get(ensg, ensg)
        sid = name_to_string_id.get(name)
        if sid:
            string_ids.add(sid)
            gene_idx_to_string_id.append((i, sid))
        else:
            gene_idx_to_string_id.append((i, None))

    edges = load_ppi_edges(links_path, string_ids, min_score=min_ppi_score)
    G = nx.Graph()
    for n in string_ids:
        G.add_node(n)
    for a, b, s in edges:
        G.add_edge(a, b, weight=s)

    # STRING ID -> cluster_id (Louvain)
    if G.number_of_nodes() == 0:
        partition = {}
    else:
        partition = community_louvain.best_partition(G)

    # string_id -> cluster_id (Louvain은 0부터 아닌 정수일 수 있음 -> 연속 재매핑)
    sid_to_cid = dict(partition)
    # 싱글톤 노드(엣지 없음)는 partition에 없을 수 있음 -> 새 클러스터 부여
    next_raw = max(partition.values(), default=-1) + 1
    for i, sid in gene_idx_to_string_id:
        if sid is None:
            continue
        if sid not in sid_to_cid:
            sid_to_cid[sid] = next_raw
            next_raw += 1

    # gene_index -> raw cluster_id (매핑 안 된 유전자는 각각 싱글톤)
    raw_cids = []
    for i, sid in gene_idx_to_string_id:
        if sid is None:
            raw_cids.append(next_raw)
            next_raw += 1
        else:
            raw_cids.append(sid_to_cid[sid])

    # 연속 0..n_clusters-1 로 재매핑
    unique = sorted(set(raw_cids))
    old_to_new = {old: new for new, old in enumerate(unique)}
    gene_to_cluster = [old_to_new[c] for c in raw_cids]
    n_clusters = len(unique)
    return gene_to_cluster, n_clusters


def aggregate_cluster_mean(x_rna: np.ndarray, gene_to_cluster: list, n_clusters: int) -> np.ndarray:
    """(N, n_genes) -> (N, n_clusters), 클러스터별 평균."""
    N, n_genes = x_rna.shape
    out = np.zeros((N, n_clusters), dtype=np.float32)
    count = np.zeros((n_clusters,), dtype=np.float32)
    for g in range(n_genes):
        c = gene_to_cluster[g]
        out[:, c] += x_rna[:, g]
        count[c] += 1.0
    for c in range(n_clusters):
        if count[c] > 0:
            out[:, c] /= count[c]
    return out


def main(args):
    # 1) 기존 HVG pkl 로드
    with open(args.rna_pkl, "rb") as f:
        data = pickle.load(f)
    case_ids = data["case_ids"]
    x_rna = np.array(data["x_rna"], dtype=np.float32)
    hvg_genes = data["hvg_genes"]
    wsi_paths = data.get("wsi_paths", [""] * len(case_ids))
    n_genes_orig = len(hvg_genes)
    print(f"[1] Loaded {args.rna_pkl}: {len(case_ids)} samples, {n_genes_orig} genes")

    # 2) ENSG -> gene_name (TSV 한 개에서)
    ensg_to_name = get_ensg_to_gene_name(args.rna_dir, args.mapping, hvg_genes)
    gene_names = [ensg_to_name.get(g, g) for g in hvg_genes]
    print(f"[2] Gene names from TSV: {len([n for n in gene_names if n != g])} mapped (rest use ENSG)")

    # 3) gene name -> STRING ID (API 또는 info 파일)
    if args.string_info and os.path.isfile(args.string_info):
        id_to_name = load_string_id_to_name_from_info(args.string_info)
        name_to_id = {v: k for k, v in id_to_name.items() if v}
        name_to_string_id = {name: name_to_id[name] for name in gene_names if name in name_to_id}
        print(f"[3] STRING ID from info file: {len(name_to_string_id)}/{len(gene_names)}")
    else:
        name_to_string_id = get_string_ids_for_genes(gene_names, species=HUMAN_TAX_ID)
        print(f"[3] STRING ID from API: {len(name_to_string_id)}/{len(gene_names)}")

    # 4) PPI 엣지 로드 (다운로드 또는 로컬)
    os.makedirs(args.string_cache, exist_ok=True)
    links_name = "9606.protein.links.v12.0.txt.gz"
    if args.string_links and os.path.isfile(args.string_links):
        links_path = args.string_links
    else:
        links_path = download_string_file(links_name, args.string_cache)
    print(f"[4] PPI links: {links_path}")

    # 5) 클러스터링
    gene_to_cluster, n_clusters = build_gene_index_to_cluster(
        hvg_genes, ensg_to_name, name_to_string_id, links_path, args.min_ppi_score
    )
    print(f"[5] Louvain clusters: n_clusters={n_clusters}")

    # 6) 클러스터별 평균으로 (N, n_clusters) 생성
    x_rna_cluster = aggregate_cluster_mean(x_rna, gene_to_cluster, n_clusters)
    print(f"[6] Aggregated x_rna shape: {x_rna_cluster.shape}")

    # cluster_genes: cluster_id -> list of original gene indices (for interpretability)
    cluster_genes = defaultdict(list)
    for g, c in enumerate(gene_to_cluster):
        cluster_genes[c].append(g)
    cluster_genes = [cluster_genes[c] for c in range(n_clusters)]

    result = {
        "case_ids": case_ids,
        "x_rna": x_rna_cluster,
        "wsi_paths": wsi_paths,
        "n_genes": n_clusters,
        "hvg_genes": hvg_genes,
        "gene_to_cluster": gene_to_cluster,
        "cluster_genes": cluster_genes,
        "n_clusters": n_clusters,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    print(f"[DONE] Saved {args.out}")
    print(f"  case_ids: {len(result['case_ids'])}, x_rna: {result['x_rna'].shape}, n_genes/n_clusters: {result['n_genes']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna_pkl", required=True, help="pre-training rna_processed.pkl 경로")
    ap.add_argument("--rna_dir", required=True, help="downloads/rnaseq 등 TSV가 있는 루트")
    ap.add_argument("--mapping", required=True, help="mapping.csv 경로")
    ap.add_argument("--out", default="./datasets/rna_ppi_clusters.pkl")
    ap.add_argument("--string_cache", default="./string_cache", help="STRING 파일 캐시 디렉터리")
    ap.add_argument("--string_links", default="", help="9606.protein.links.v12.0.txt.gz 경로 (없으면 다운로드)")
    ap.add_argument("--string_info", default="", help="9606.protein.info.v12.0.txt.gz (선택, 없으면 API로 매핑)")
    ap.add_argument("--min_ppi_score", type=int, default=150)
    args = ap.parse_args()
    main(args)
