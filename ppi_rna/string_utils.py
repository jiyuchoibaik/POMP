"""
STRING DB 연동 유틸: 유전자명 → STRING ID 매핑, PPI 엣지 로드.
"""
import os
import gzip
import time
import requests
from typing import Dict, List, Set, Tuple

STRING_API = "https://string-db.org/api"
STRING_VERSION = "12.0"
STRING_DOWNLOAD_BASE = "https://stringdb-downloads.org/download"
HUMAN_TAX_ID = 9606


def get_string_ids_for_genes(
    gene_names: List[str],
    species: int = HUMAN_TAX_ID,
    batch_size: int = 100,
    caller_identity: str = "POMP_ppi_rna",
) -> Dict[str, str]:
    """
    STRING API get_string_ids로 유전자명 → STRING ID 매핑.
    gene_names에 없는 경우 또는 API 실패 시 해당 키는 반환하지 않음.
    Returns: { gene_name: string_id } (예: {"TP53": "9606.ENSP00000269305"})
    """
    result = {}
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i : i + batch_size]
        params = {
            "identifiers": "\r".join(batch),
            "species": species,
            "limit": 1,
            "echo_query": 1,
            "caller_identity": caller_identity,
        }
        try:
            r = requests.post(
                f"{STRING_API}/tsv-no-header/get_string_ids",
                data=params,
                timeout=60,
            )
            r.raise_for_status()
            for line in r.text.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    query_item = parts[0].strip()
                    string_id = parts[2].strip()
                    result[query_item] = string_id
        except Exception as e:
            print(f"[WARN] STRING get_string_ids batch failed: {e}")
        time.sleep(1.0)
    return result


def load_string_id_to_name_from_info(info_path: str) -> Dict[str, str]:
    """
    9606.protein.info.v12.0.txt.gz 파싱: string_id -> preferred_name (gene symbol).
    TSV: string_protein_id, preferred_name, ...
    """
    id_to_name = {}
    open_fn = gzip.open if info_path.endswith(".gz") else open
    with open_fn(info_path, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id_to_name[parts[0]] = parts[1]
    return id_to_name


def load_ppi_edges(
    links_path: str,
    string_ids: Set[str],
    min_score: int = 150,
) -> List[Tuple[str, str, int]]:
    """
    STRING protein.links 파일에서 우리 string_ids 집합에 속한 엣지만 로드.
    links 포맷: protein1, protein2, score (0-1000).
    Returns: [(protein1, protein2, score), ...]
    """
    edges = []
    open_fn = gzip.open if links_path.endswith(".gz") else open
    with open_fn(links_path, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            a, b, score = parts[0], parts[1], int(parts[2])
            if score < min_score:
                continue
            if a in string_ids and b in string_ids:
                edges.append((a, b, score))
    return edges


def download_string_file(
    filename: str,
    save_dir: str,
    base_url: str = STRING_DOWNLOAD_BASE,
) -> str:
    """
    STRING 다운로드 페이지에서 파일명으로 URL 구성해 다운로드.
    예: 9606.protein.links.v12.0.txt.gz -> protein.links.v12.0/9606.protein.links.v12.0.txt.gz
    """
    os.makedirs(save_dir, exist_ok=True)
    # folder = filename에서 9606. 제거 후 .txt.gz 제거 -> protein.links.v12.0
    folder = filename.replace("9606.", "").replace(".txt.gz", "")
    url = f"{base_url}/{folder}/{filename}"
    path = os.path.join(save_dir, filename)
    if os.path.isfile(path):
        return path
    print(f"[STRING] Downloading {url} ...")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return path
