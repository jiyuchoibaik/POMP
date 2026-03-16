# PPI 기반 RNA 클러스터 표현

HVG 2000개 raw 발현 대신, STRING PPI 네트워크로 클러스터링한 뒤 클러스터별 평균을 RNA 표현으로 사용합니다.

## 1. 전처리 (PPI 클러스터링)

기존 `pre-training/datasets/rna_processed.pkl`과 동일한 데이터 소스(rna_dir, mapping)가 필요합니다.

```bash
cd ppi_rna

# 예: ppi_rna 폴더에서 실행 (데이터가 pre-training/datasets, downloads에 있을 때)
python preprocess_ppi_rna.py \
  --rna_pkl ../pre-training/datasets/rna_processed.pkl \
  --rna_dir ../pre-training/downloads/rnaseq \
  --mapping ../pre-training/downloads/mapping.csv \
  --out ./datasets/rna_ppi_clusters.pkl \
  --string_cache ./string_cache
```

- `--string_links`: 이미 받은 `9606.protein.links.v12.0.txt.gz` 경로를 주면 재다운로드 생략
- `--string_info`: (선택) `9606.protein.info.v12.0.txt.gz` 경로를 주면 유전자명→STRING ID 매핑을 API 대신 파일로 수행

출력 `rna_ppi_clusters.pkl`에는 `x_rna`가 `(N, n_clusters)` 형태로 들어갑니다.

## 2. 학습

```bash
cd ppi_rna

python main_multimodal_pretrain_ppi.py \
  --data_pkl ./datasets/rna_ppi_clusters.pkl \
  --output_dir ./output_pretrain_ppi \
  --log_dir ./output_pretrain_ppi
```

`rna_dim`은 pkl의 `n_genes`(= n_clusters)를 자동으로 사용합니다. `--n_genes`를 주면 그 값으로 덮어씁니다.

## 의존성

- `networkx`, `python-louvain` (또는 `louvain`), `requests`

```bash
pip install networkx python-louvain requests
```
