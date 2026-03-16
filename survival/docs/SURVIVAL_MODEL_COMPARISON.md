# Survival 모델 구조 비교: 예전(linear/cosine) vs 현재(risk_head)

## 1. 예전 실험 (log-test1-2026-03-08, checkpoint-tcga_luad-500.pth)

### 모델 (git 8cfcdab 기준)

| 구성요소 | 예전 이름/역할 | 비고 |
|----------|----------------|------|
| RNA → embed | `self.linear` = `nn.Linear(n_genes, embed_dim)` | 현재 `rna_linear`와 동일 역할 |
| 생존 예측 헤드 | `self.risk_head` = `Sequential(Linear(embed_dim, 1), Sigmoid())` | path_guided 쪽에서만 사용 |
| 추가 | `GlobalAttentionPooling` (GAP) | 현재 모델에는 없음 |

### forward / 출력

- **forward_features** 반환: `corr, img, X_omics`
  - `corr` = **cosine_similarity(img_cls, omics_cls)** → 학습 파라미터 없음, 이미지·오믹스 CLS 유사도
- **path_guided_omics_encoder** 반환: `risk_head(path_guid[:, 0:1, :])` → **path-guided 후 risk_head 하나만** 사용
- **엔진 조합**: `outputs = outputs1 + outputs2`
  - **outputs1** = `corr` (cosine, unsqueeze(1) → (B,1))
  - **outputs2** = path_guided risk

즉, 예전에는 **outputs1이 risk_head가 아니라 cosine similarity**이고, **학습되는 risk 경로는 path_guided + risk_head 한 개**뿐이었음.

---

## 2. 현재 실험 (z-score pre-train, checkpoint-tcga_luad_v2-400.pth)

### 모델 (현재 survival/model/models_pomp.py)

| 구성요소 | 현재 이름/역할 | 비고 |
|----------|----------------|------|
| RNA → embed | `self.rna_linear` | pre-training과 동일 (체크포인트 호환) |
| 생존 예측 헤드 | `self.risk_head` = `Sequential(Linear(embed_dim, 1), Sigmoid())` | **두 군데** 사용 |
| pre-train 호환 | `pom_head`, `mom_head`, `logit_scale` 등 | 예전 survival 모델에는 없었음 |

### forward / 출력

- **forward_features** 반환: `img_cls, omics_cls, img, omics_inp`
- **forward** 반환: `risk1, img, omics_inp`
  - **risk1** = `risk_head(omics_cls)` → **오믹스만으로 학습 가능한 risk**
- **path_guided_omics_encoder** 반환: `risk_head(fused[:, 0:1, :])` → 동일 risk_head, fusion 입력
- **엔진 조합**: `outputs = outputs1 + outputs2`
  - **outputs1** = risk_head(omics_cls)  (B,1)
  - **outputs2** = path_guided risk_head  (B,1)

즉, 현재는 **outputs1·outputs2 모두 risk_head를 통한 학습 가능한 risk**이고, **학습되는 risk 경로가 두 개**임.

---

## 3. 차이 요약

| 항목 | 예전 | 현재 |
|------|------|------|
| outputs1 | **cosine(img_cls, omics_cls)** (비학습) | **risk_head(omics_cls)** (학습) |
| outputs2 | risk_head(path_guided) | risk_head(path_guided) |
| 학습 risk 경로 수 | 1개 (path_guided만) | 2개 (omics_cls + path_guided) |
| RNA 레이어 이름 | `linear` | `rna_linear` (pre-train과 동일) |
| GAP | 있음 | 없음 |
| Pre-train 체크포인트 | head/linear 등 다른 구조 | rna_linear, pom_head, mom_head 등 동일 |

---

## 4. 성능 차이 해석

- 예전: **한 개의 학습 risk 경로(path_guided)** + **cosine으로 이미지–오믹스 정렬 신호** → val c-index ~0.65
- 현재: **두 개의 학습 risk 경로** + pre-train과 완전 동일 백본 → val c-index ~0.55

가능한 원인:

1. **outputs1이 cosine일 때**  
   이미지–오믹스 정렬이 이미 pre-train에서 학습되어 있어, 그대로 cosine을 쓰는 것이 risk_head(omics_cls)보다 더 안정적이었을 수 있음.
2. **risk_head(omics_cls)만으로는**  
   오믹스 단일 브랜치가 작은 survival 데이터에서 과적합되거나 스케일이 달라 path_guided와 더했을 때 불리할 수 있음.

---

## 5. 현재 코드 동작

Survival 코드는 **예전과 동일한 출력 방식**으로 맞춰 두었음:

- **outputs1** = cosine(img_cls, omics_cls)
- **outputs2** = path_guided risk_head

즉, 학습되는 risk 경로는 path_guided + risk_head 한 개뿐이고, outputs1은 비학습 cosine 신호만 사용함.
