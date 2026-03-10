# Survival 파인튜닝 문서 (POMP)

Pre-training된 POMP 모델을 생존 분석(Cox proportional hazards) 태스크로 파인튜닝하는 파이프라인 정리.

---

## 1. 개요

- **목적:** WSI + RNA-seq 표현을 이용해 **생존 시간(survival)** 을 예측하고, **위험도(risk score)** 로 고/저위험 그룹을 구분.
- **데이터:** TCGA-LUAD 5-fold cross-validation (train / validation / test).
- **입력:** (1) WSI 패치 집합 `(max_num_region, 3, 256, 256)`, (2) RNA 벡터 `(n_genes,)`, (3) censored, (4) survival time.
- **출력:** risk score → C-index, log-rank p-value로 평가.

---

## 2. 폴더/파일 구조

```
survival/
├── main_multimodal_survival.py   # 학습·평가 진입점 (5-fold, early stopping, --eval)
├── engine_multimodal_survival.py # 1 에폭 학습, evaluate (Cox loss, c-index, p-value)
├── model/
│   ├── models_pomp.py            # ViT + risk_head + path_guided_omics_encoder
│   └── cox_loss.py               # PartialLogLikelihood, calc_concordance_index, cox_log_rank
├── utils/
│   ├── data_loader.py            # POMPDataset (split, max_num_region)
│   ├── options.py                # get_args_parser_finetune, create_logger
│   ├── misc.py, lr_sched.py, lr_decay.py, pos_embed.py
│   └── ...
└── datasets/
    └── build_survival_pkl.py     # luad_cv_splits.pkl 생성 (train/val/test, x_rna, censored, survival)
```

---

## 3. 데이터 파이프라인

### 3.1 전제 조건

- **pkl:** `datasets/luad_cv_splits.pkl` (또는 `--data_dir` 로 지정).
  - `build_survival_pkl.py` 로 생성.
  - 구조: `{ fold_idx: { "train": {...}, "validation": {...}, "test": {...} } }`.
  - 각 split: `region_pixel_5x` (WSI 경로), `x_rna` (N × n_genes), `censored`, `survival`.

### 3.2 데이터셋 (utils/data_loader.py)

- **클래스:** `POMPDataset(data, split, max_num_region=250)`.
- **반환:** `(regions, X_rna, censored, survival)`  
  - regions: `(max_num_region, 3, 256, 256)` (부족하면 반복, 많으면 앞에서 자름).

### 3.3 주요 학습/평가 옵션 (main에서 추가된 인자 포함)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `./datasets/lihc_cv_splits.pkl` | 5-fold pkl 경로 (LUAD는 luad_cv_splits.pkl 등) |
| `--n_genes` | 2000 | RNA 차원 (pre-training·pkl과 동일) |
| `--max_num_region` | 250 | 샘플당 최대 패치 수 (메모리 절약) |
| `--finetune` | (기본 체크포인트) | Pre-training 체크포인트 경로 |
| `--epochs` | 80 | 최대 에폭 (early stopping으로 조기 종료) |
| `--batch_size` | 1 | 배치 크기 |
| `--accum_iter` | 10 | gradient accumulation |
| `--lr` / `--blr` | blr 5e-4 | 학습률 (lr 미지정 시 blr 사용) |
| `--warmup_epochs` | 8 | warmup 에폭 |
| `--num_workers` | 10 | DataLoader workers |
| `--prefetch_factor` | 4 | DataLoader prefetch (num_workers>0일 때) |
| `--output_dir` | `./output_finetune_dir/ablation_loss` | best 모델·예측·체크포인트 저장 |
| `--log_dir` | (output_dir과 동일) | TensorBoard 로그 |
| `--save_every` | 100 | N 에폭마다 checkpoint-{exptype}-fold{k}-epoch{N}.pth 저장 (0이면 비활성) |
| `--eval` | False | True면 학습 없이 각 fold best 모델 로드 후 test만 수행 |
| `--device` | cuda | cuda / cpu |
| `--gradient_checkpointing` | True | OOM 완화 (--no_gradient_checkpointing 으로 끔) |

---

## 4. 모델 구조 (survival 쪽)

- **백본:** Pre-training과 동일한 ViT + RNA linear + path_guided_omics_encoder.
- **Survival 전용:**  
  - `risk_head`: Linear(embed_dim → 1) → risk score.  
  - forward: `outputs1 = risk_head(image_embed)` + `outputs2 = path_guided_omics_encoder(image_embed, omics_embed)` → **outputs = outputs1 + outputs2** (Cox loss 입력).
- Pre-training 체크포인트 로드 시 `head`, `linear`(shape 일치 시), path_guided 등은 로드하고, `risk_head` 는 제거 후 새로 초기화.

---

## 5. 학습 목표 (engine_multimodal_survival.py)

- **손실:** Cox partial log-likelihood (Negative, 최소화).
- **정렬:** survival 시간 내림차순 정렬 후 loss 계산 (Cox 요구사항).
- **Mixed precision:** `torch.amp.autocast("cuda")` 사용.
- **Gradient:** `accum_iter` 만큼 모은 뒤 한 번에 업데이트.

---

## 6. 평가 지표

- **C-index (Concordance index):** 순위 일치도. 1에 가까울수록 “위험도가 높은 사람이 더 일찍 이벤트” 순서를 잘 맞춤.
- **p-value (log-rank):** risk score 중위수로 고/저위험 그룹 나눈 뒤, 두 그룹의 생존 곡선 차이가 통계적으로 유의한지 검정. p < 0.05 이면 유의.

---

## 7. 학습 흐름 (main_multimodal_survival.py)

1. **5-fold:** `data_cv_splits` 의 각 fold에 대해 동일 설정으로 학습.
2. **Early stopping:** validation c-index가 **연속 5 에폭** 동안 best를 갱신하지 않으면 해당 fold 종료.
3. **Best 모델:** validation c-index 최대인 에폭의 모델을 `best_model_{exptype}_fold{k}.pth` 로 저장.
4. **주기 저장:** `save_every > 0` 이면 N 에폭마다 `checkpoint-{exptype}-fold{k}-epoch{N}.pth` 추가 저장.
5. **TensorBoard:** `log_dir` 에 train/val/test c-index 등 기록.
6. **예측 저장:** best 시점의 test 예측을 `predict_result_{exptype}/predict_kfold_{k}.csv` 에 저장하고, 학습 종료 후 `predict_result.csv` 로 합침.

---

## 8. 실행 방법

### 8.1 학습 (예: GPU 0, LUAD 5-fold)

```bash
cd /path/to/POMP
CUDA_VISIBLE_DEVICES=0 python survival/main_multimodal_survival.py \
  --data_dir ./survival/datasets/luad_cv_splits.pkl \
  --n_genes 2000 \
  --max_num_region 250 \
  --finetune ./pre-training/output_pretrain/checkpoint-tcga_luad-500.pth \
  --epochs 80 \
  --accum_iter 10 \
  --num_workers 24 \
  --output_dir ./output_finetune \
  --log_dir ./output_finetune \
  --exptype tcga_luad
```

### 8.2 평가만 (이미 학습된 best 모델로 test c-index/p-value 출력)

```bash
CUDA_VISIBLE_DEVICES=0 python survival/main_multimodal_survival.py \
  --data_dir ./survival/datasets/luad_cv_splits.pkl \
  --n_genes 2000 \
  --max_num_region 250 \
  --finetune ./pre-training/output_pretrain/checkpoint-tcga_luad-500.pth \
  --output_dir ./output_finetune \
  --exptype tcga_luad \
  --eval
```

- 각 fold의 `best_model_tcga_luad_fold{k}.pth` 가 `output_dir` 에 있어야 하며, 없으면 해당 fold는 스킵되고 경고 로그만 남음.
- 마지막에 **평가 모드 요약:** test c-index (fold별 및 mean ± std), test p-value (fold별) 출력.

---

## 9. 실제 실험 결과 (현 시점 정리)

- **실험:** `output_finetune_dir/ablation_loss/log-test1-2026-03-08 23:55:34.log`  
  - 데이터: LUAD 5-fold, pre-training: `checkpoint-tcga_luad-500.pth`, `max_num_region` 250, gradient checkpointing 등 적용.

### 9.1 수치 요약

| 구분 | 값 |
|------|-----|
| **Train c-index (fold별 평균)** | 0.6171 (fold별: 0.6435, 0.6108, 0.6168, 0.6339, 0.5805) |
| **Val c-index** | 0.6470 ± 0.0342 (fold별: 0.6486, 0.6555, 0.6975, 0.6431, 0.5905) |
| **Test c-index** | 0.6470 ± 0.0342 (동일) |
| **총 학습 시간** | 5:54:51 |

- Val/Test가 동일한 이유: 현재 코드에서 validation set과 test set이 동일한 데이터로 사용된 것으로 보임 (같은 DataLoader로 평가).

### 9.2 해석

- **C-index ≈ 0.65:** 무작위(0.5)보다 명확히 나음. “위험도 순서”에 대한 예측 능력이 있다고 볼 수 있음.
- **다만 0.7~0.8 수준의 정교한 생존 예측**에는 미치지 못하는 수치. 데이터 크기, 패치 수, 학습 설정, 모델 용량 등 개선 여지가 있음.
- **p-value:** fold·에폭마다 로그에 기록됨. 고/저위험 그룹 차이가 유의한 fold도 있고 아닌 fold도 있음 (해당 로그에서 fold별 best 시점 p-value 확인 가능).

---

## 10. 출력 파일 정리

| 경로 | 내용 |
|------|------|
| `output_dir/best_model_{exptype}_fold{0..4}.pth` | fold별 best 모델 가중치 |
| `output_dir/predict_result_{exptype}/predict_kfold_{k}.csv` | fold별 예측 (predict, censored, survival) |
| `output_dir/predict_result_{exptype}/predict_result.csv` | 5-fold 예측 합친 결과 (학습 완료 시) |
| `output_dir/checkpoint-{exptype}-fold{k}-epoch{N}.pth` | save_every 설정 시 N 에폭마다 저장 |
| `output_dir/log-{exptype}-{timestamp}.log` | options.py의 create_logger가 쓰는 로그 파일 (output_dir 기준) |
| `log_dir/` | TensorBoard events |

---

## 11. 참고 (Pre-training과의 연결)

- Pre-training에서 학습한 **이미지·오믹스 표현**을 고정하지 않고, survival loss로 **전체를 파인튜닝**.
- `--finetune` 에 pre-training 체크포인트를 주면, 호환되는 키만 로드하고 `risk_head` 등 survival 전용 파라미터는 새로 초기화한 뒤 함께 학습.

---

*문서 기준: 현재 코드 (main_multimodal_survival.py, engine_multimodal_survival.py, model/models_pomp.py, model/cox_loss.py, utils/data_loader.py, utils/options.py) 및 실험 로그 (log-test1-2026-03-08 23:55:34.log).*
