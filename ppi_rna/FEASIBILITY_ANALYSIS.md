# PPI 기반 RNA 클러스터 표현 — 실현 가능성 분석

## 요약: **가능합니다**

HVG 2000개 raw 발현값을 PPI 네트워크 기반 기능적 클러스터의 대표값(평균)으로 바꾸는 것은 **기술적으로 실현 가능**하며, 기존 파이프라인(`rna_dim`만 `n_clusters`로 바꾸면 됨)과의 호환도 좋습니다.

---

## 1. 현재 파이프라인 정리

| 단계 | 내용 |
|------|------|
| **입력** | TCGA-LUAD RNA-seq (STAR Counts TSV), `gene_id`(ENSG), `gene_name` 사용 가능 |
| **전처리** | `preprocess_rna.py`: HVG 2000 선택 → log1p 정규화 → `x_rna` (N, 2000) |
| **모델** | `models_pomp.py`: `rna_linear = nn.Linear(rna_dim, embed_dim)`, `mom_head = nn.Linear(embed_dim, rna_dim)` |
| **사용처** | `x_rna`는 `(B, rna_dim)`으로만 사용되므로, 차원만 맞으면 구조 변경 최소화 가능 |

즉, **`rna_dim=2000` → `rna_dim=n_clusters`**로만 바꾸면 모델 구조 변경을 최소화할 수 있습니다.

---

## 2. 제안 방식의 실현 가능성

### 2.1 PPI 네트워크 소스 (STRING DB)

- **STRING DB** (https://string-db.org) 사용 가능.
- 제공 내용:
  - **API**: `/api/tsv/get_string_ids?` — gene name 등 → STRING ID 매핑
  - **API**: `/api/tsv/network?` — 특정 유전자들에 대한 상호작용 네트워크 (엣지 리스트)
  - **전체 다운로드**: organism별 `protein.links.*.txt.gz` 등으로 전체 PPI 다운로드 가능
- Human (NCBI 9606) 지원, gene name / Ensembl 계열 ID 매핑 가능.

→ **HVG 2000개 유전자를 STRING에 맞춰 매핑하고, 이들만으로 서브네트워크를 만드는 것은 가능**합니다.

### 2.2 유전자 ID 매핑

- 현재 데이터: STAR Counts TSV에 `gene_id`(ENSG), `gene_name` 존재.
- STRING API: `get_string_ids`에 **gene name** 또는 다른 ID를 넣으면 STRING ID로 변환.
- 선택지:
  1. **gene_name**으로 STRING API 호출 (권장: 구현 단순)
  2. STRING/Ensembl 매핑 파일 다운로드 후 ENSG ↔ STRING ID 매핑

→ **2000개 HVG를 STRING 네트워크의 노드로 쓸 수 있는 매핑 경로가 있습니다.** (일부 미매핑은 아래처럼 처리 가능)

### 2.3 PPI 기반 클러스터링

- 방법: HVG 2000개 중 STRING에 매핑된 유전자들로 **서브네트워크** 구성 → **커뮤니티 탐지**로 클러스터 부여.
- 도구 예:
  - **Louvain**, **Leiden** (예: `python-louvain`, `leidenalg`, 또는 `networkx` + 간단 구현)
  - 또는 STRING에서 제공하는 **precomputed cluster** (있는 경우) 활용
- 결과: 각 유전자 → `cluster_id`. 클러스터 개수 = `n_clusters` (고정).

→ **“PPI로 유전자를 기능적 클러스터로 묶는다”는 단계는 표준적인 네트워크 분석으로 구현 가능**합니다.

### 2.4 클러스터별 대표값 (평균) → `x_rna: (B, n_clusters)`

- 샘플별로, 각 클러스터에 속한 유전자들의 (이미 log1p 등으로 정규화된) 발현값의 **평균**을 계산.
- 한 클러스터에 유전자가 1개뿐이면 그 유전자 값이 대표값.
- 최종: **샘플 × 클러스터** 행렬 → `(B, n_clusters)`.
- max 등 다른 집계도 동일한 방식으로 적용 가능.

→ **구현 난이도 낮고, 기존 `(B, rna_dim)` 인터페이스와 형태만 맞추면 됨.**

### 2.5 매핑 실패/빈 클러스터 처리

- **STRING에 없는 유전자**:  
  - (A) 해당 유전자만으로 **싱글톤 클러스터** 1개씩 부여하거나,  
  - (B) 제외하고 클러스터링에만 참여시키지 않음 (대표값 계산 시 해당 유전자 무시).
- **엣지가 없어 서브네트워크가 비연결**:  
  - 연결 성분별로 클러스터 부여하거나,  
  - 싱글톤/작은 성분은 각각 별도 클러스터로 두면 됨.

→ **미매핑·희소 네트워크도 정책만 정하면 처리 가능**합니다.

---

## 3. 기존 코드 유지 + 새 코드만 분리

- **기존**: `pre-training/datasets/preprocess_rna.py`, `pre-training/model/models_pomp.py`, survival 등 **그대로 유지**.
- **새 작업만** 별도 디렉터리(예: `ppi_rna/`)에 두는 방식이면:
  - `ppi_rna/`: PPI 클러스터링 + 클러스터 대표값 계산 **전처리 스크립트**
  - 출력: `x_rna`가 `(N, n_clusters)`인 pkl (및 `cluster_info`, `n_clusters` 등 메타데이터)
  - 학습/추론 시에는 **기존 데이터 대신** 이 pkl을 쓰고, `rna_dim=n_clusters`만 넘기면 됨.

→ **기존 코드를 건드리지 않고, 새 폴더에서만 “대체 전처리 + 설정값 변경”으로 적용 가능**합니다.

---

## 4. 정리

| 항목 | 판단 |
|------|------|
| STRING DB로 PPI 가져오기 | ✅ 가능 (API 또는 bulk 다운로드) |
| HVG 2000 → STRING 노드 매핑 | ✅ 가능 (gene name 또는 매핑 파일) |
| PPI 서브네트워크로 클러스터링 | ✅ 가능 (Louvain/Leiden 등) |
| 클러스터 평균 → (B, n_clusters) | ✅ 가능 |
| 모델 변경 최소화 (rna_dim만 변경) | ✅ 가능 |
| 기존 코드 유지 + 새 폴더로 분리 | ✅ 가능 |

**결론: 제안하신 “RNA를 PPI 네트워크 기반 클러스터 표현으로 바꾸는 것”은 가능한 이야기이며, 현재 구조를 크게 바꾸지 않고 새 전처리 파이프라인만 추가하는 형태로 구현할 수 있습니다.**

이어서 `ppi_rna/` 아래에 전처리 스크립트 골격(STRING 연동, 클러스터링, 클러스터 평균 계산)을 만들 수 있습니다. 원하시면 그 단계부터 구체적인 코드 초안을 제안하겠습니다.
