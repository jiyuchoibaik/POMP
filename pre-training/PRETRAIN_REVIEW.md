# Pre-training 코드 점검 요약

## 1. POC 불안정 (epoch 15에서 4.48→16에서 4.81로 튐)

**원인**  
- `engine_multimodal_pretrain_2.py`에서 POC용 feature를 **detach 없이** 50스텝 전부 누적하고 있음.  
- 50개 forward 그래프가 한 번에 backward에 참여하면서 gradient가 불안정해짐.

**조치**  
- `engine_multimodal_pretrain.py`처럼 **마지막 accumulation 스텝만 grad 연결**하도록 복구.  
- 즉, `k < accum_iter - 1`일 때는 `img_cls.squeeze(1).detach()`, `omics_cls_poc.squeeze(1).detach()`로 append.

---

## 2. Gradient clipping 미적용

**원인**  
- `train_one_epoch(..., max_norm=args.clip_grad)`로 받지만, `loss_scaler(loss, optimizer, ...)` 호출 시 **clip_grad 인자를 넘기지 않음**.  
- `utils/misc.py`의 `NativeScaler.__call__`는 `clip_grad`가 있을 때만 `clip_grad_norm_`을 수행.

**조치**  
- `loss_scaler(loss, optimizer, parameters=..., clip_grad=max_norm, update_grad=...)` 형태로 호출하도록 수정.

---

## 3. POM이 ~0.637에 정체

- 0.637 ≈ ln(2)에 가깝고, 이진 분류에서 무작위에 가까움.  
- POC가 안정화되면 POM gradient 비중이 상대적으로 살아날 수 있음.  
- 우선 POC detach + clip_grad 적용 후 재학습해 보고, 여전히 개선 없으면 `loss_pom` 가중치(6.0) 상향이나 POM만의 gradient scale 검토 가능.

---

## 4. MOM

- 19 → 10으로 정상 감소.  
- `mom_scale`, `mom_weight` 설정은 유지.

---

## 5. 기타

- **path_guided_omics_encoder**: MOM/POM은 배치 한 번에 호출하는 쪽이 `engine_multimodal_pretrain.py`에만 있고, `engine_multimodal_pretrain_2.py`는 여전히 for-loop으로 100번 호출.  
- 학습 속도 개선을 위해 engine_2에서도 배치 호출로 바꿀 수 있음 (선택).
