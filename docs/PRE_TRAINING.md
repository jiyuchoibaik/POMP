# Pre-training 문서 (POMP)

TCGA-LUAD WSI + RNA-seq 기반 멀티모달 사전학습 파이프라인 정리.

---

## 1. 개요

- **목적:** WSI(패치)와 RNA-seq을 함께 보고, 이미지–오믹스 정렬·마스킹 복원·매칭을 학습해 **survival 파인튜닝용 표현**을 얻음.
- **데이터:** TCGA-LUAD (case별 WSI regions.npy + RNA-seq 벡터).
- **입력:** (1) WSI 패치 집합 `(N_patch, 3, 256, 256)`, (2) RNA 벡터 `(n_genes,)` (기본 2000).

---

## 2. 폴더/파일 구조

```
pre-training/
├── main_multimodal_pretrain.py   # 학습 진입점
├── engine_multimodal_pretrain.py # 1 에폭 학습 (POC, MOM, POM loss)
├── model/
│   └── models_pomp.py            # ViT + RNA encoder + path_guided_omics_encoder
├── utils/
│   ├── data_loader.py            # TCGALUADDataset, build_dataset
│   ├── misc.py                   # 저장/로드, 분산 등
│   ├── lr_sched.py               # 학습률 스케줄
│   ├── lr_decay.py               # layer-wise decay
│   └── pos_embed.py              # positional embedding 보간
└── datasets/
    ├── preprocess_rna.py        # RNA 전처리 → rna_processed.pkl
    ├── extract_patches.py       # WSI → regions.npy
    ├── download_multimodal.py   # 데이터 다운로드
    └── ...
```

---

## 3. 데이터 파이프라인

### 3.1 전제 조건

- **RNA:** `preprocess_rna.py` 출력 `rna_processed.pkl`  
  - 키: `case_ids`, `x_rna` (N × n_genes, log1p 등 정규화), `n_genes`, (필요 시) `wsi_paths`.
- **WSI:** 케이스별 `regions.npy` (패치 배열, shape `(K, 3, 256, 256)`).

### 3.2 데이터셋 (utils/data_loader.py)

- **클래스:** `TCGALUADDataset(data, max_num_region=300)`.
- **data:** `case_ids`, `x_rna`, `wsi_paths`, `n_genes` 포함 dict.
- **반환:** `(regions, x_rna)`  
  - `regions`: `(max_num_region, 3, 256, 256)` (부족하면 반복 샘플링, 많으면 무작위 선택).

### 3.3 학습 설정 (main_multimodal_pretrain.py 기본값)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data_pkl` | `./datasets/rna_processed.pkl` | RNA + 메타데이터 pkl |
| `--n_genes` | 2000 | RNA 차원 (전처리와 동일) |
| `--max_patches` | 300 | 샘플당 최대 패치 수 |
| `--epochs` | 501 | 총 에폭 |
| `--batch_size` | 1 | 배치 크기 |
| `--accum_iter` | 50 | gradient accumulation → effective batch 50 |
| `--mask_ratio` | 0.3 | MOM용 유전자 마스킹 비율 |
| `--lr` | 5e-4 | 학습률 |
| `--warmup_epochs` | 50 | warmup 에폭 |
| `--weight_decay` | 1e-5 | AdamW weight decay |
| `--layer_decay` | 0.75 | layer-wise lr decay |
| `--drop_path` | 0.1 | stochastic depth |
| `--save_every` | 100 | 체크포인트 저장 주기 (에폭 단위) |
| `--output_dir` | `./output_pretrain` | 체크포인트·로그 저장 경로 |

- **저장 조건:** `epoch >= 50` 이고 `epoch % save_every == 0` 일 때만 저장 → **100, 200, 300, 400, 500** 에폭에 저장됨.

---

## 4. 모델 구조 (model/models_pomp.py)

- **백본:** ViT (timm) 기반, `vit_base_patch16` (img 256, patch 16, embed_dim 384, depth 6 등).
- **변경점:**  
  - RNA 1종만 사용 (NUM_OMICS=1).  
  - `rna_linear`: (n_genes → embed_dim).  
  - `path_guided_omics_encoder`: 이미지 embedding으로 오믹스 쿼리 → cross-attn + fuse_transf → **POM head(2-class)**, **MOM head(rna_dim 복원)**.
- **출력:**  
  - `forward`: `img_cls`, `omics_cls` (contrastive용), `image_embed`, `omics_embed` (path_guided 입력).  
  - path_guided: 마스크된 RNA 복원 로짓, POM 이진 로짓.

---

## 5. 학습 목표 (engine_multimodal_pretrain.py)

매 `accum_iter` 스텝마다:

1. **POC (Pathology-Omics Contrastive)**  
   - 이미지/오믹스 cls embedding 정규화 후 대조 학습 (대각선 positive, cross-entropy).  
   - `loss_poc`.

2. **MOM (Masked Omics Modeling)**  
   - 유전자 `mask_ratio`만큼 마스킹 → path_guided로 복원.  
   - MSE: `loss_mom`.

3. **POM (Pathology-Omics Matching)**  
   - positive: 같은 샘플 이미지–오믹스 쌍.  
   - negative: 다른 샘플 이미지–오믹스 조합 (유사도 기반 샘플링).  
   - 이진 분류 cross-entropy: `loss_pom`.

**총 손실:**  
`loss = 1.0 * loss_poc + 6.0 * loss_pom + 3.0 * loss_mom`

- Mixed precision (AMP), gradient scaling 사용.

---

## 6. 실행 방법

```bash
cd /path/to/POMP
# 단일 GPU
python pre-training/main_multimodal_pretrain.py \
  --data_pkl ./pre-training/datasets/rna_processed.pkl \
  --n_genes 2000 \
  --max_patches 300 \
  --epochs 501 \
  --accum_iter 50 \
  --mask_ratio 0.3 \
  --save_every 100 \
  --output_dir ./output_pretrain \
  --log_dir ./output_pretrain
```

- 재개: `--resume ./output_pretrain/checkpoint-tcga_luad-400.pth` 등.

---

## 7. 저장되는 것

- **경로:** `args.output_dir` (기본 `./output_pretrain`).
- **파일:**  
  - `checkpoint-{exptype}-{epoch}.pth` (예: `checkpoint-tcga_luad-500.pth`)  
  - 내부: `model`(state_dict), `optimizer`, `epoch`, `scaler`, `args` 등.
- **로그:** `log_pretrain_{exptype}.txt`, TensorBoard (`log_dir`).

---

## 8. 참고 (survival과의 연결)

- Survival 파인튜닝 시 `--finetune pre-training/output_pretrain/checkpoint-tcga_luad-500.pth` 로 이 체크포인트를 불러옴.
- Pre-training에는 classification/survival head가 없으므로, survival 모델의 `head`, `linear`, `risk_head` 등은 새로 초기화되거나 survival 쪽에서 정의함.

---

*문서 기준: 현재 코드 (main_multimodal_pretrain.py, engine_multimodal_pretrain.py, model/models_pomp.py, utils/data_loader.py).*
