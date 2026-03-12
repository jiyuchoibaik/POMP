# engine_multimodal_pretrain_2.py  ─ POC는 마스킹 안 한 RNA, loss 스케일 맞춤
# 기존 engine_multimodal_pretrain.py 와 동일한 구조 + 아래 변경만 적용.
#  - POC: omics_cls를 마스킹 안 한 x_rna로 한 번 더 forward 해서 사용 (정렬 학습 강화).
#  - MOM loss 스케일: 1/sqrt(n_genes) 로 보정해 gradient 균형 (POC/POM이 실제로 학습되도록).
#  - [수정1] detach 제거 → 모든 샘플에서 gradient 흐름
#  - [수정2] loss_scaler에 clip_grad 실제로 전달 (이전엔 누락되어 clip_grad가 적용 안 됨)
import math
import sys
import random
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched
import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    accum_iter = args.accum_iter
    mask_ratio = getattr(args, "mask_ratio", 0.3)

    optimizer.zero_grad()
    if log_writer:
        print(f"log_dir: {log_writer.log_dir}")

    # [수정1] detach 없이 전부 grad 연결 → 단순 리스트로 누적
    image_feats_list = []
    omics_feats_list = []
    image_embeds = []
    omics_embeds = []
    rna_masks    = []
    k = -1

    for data_iter_step, (regions, x_rna) in enumerate(data_loader):

        if (data_iter_step + 1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        regions = regions.to(device, non_blocking=True)
        x_rna   = x_rna.to(device, non_blocking=True)

        rna_orig     = x_rna.clone()
        n_genes      = x_rna.shape[1]
        n_mask       = int(n_genes * mask_ratio)
        mask_idx     = torch.randperm(n_genes)[:n_mask]
        x_rna_masked = x_rna.clone()
        x_rna_masked[:, mask_idx] = 0.0

        rna_masks.append(rna_orig)

        with torch.cuda.amp.autocast():
            # Forward 1: 마스킹된 RNA → MOM/POM용 image_embed, omics_embed, img_cls
            samples_masked = [regions, x_rna_masked]
            img_cls, _, image_embed, omics_embed = model(samples_masked)

            # Forward 2: 마스킹 안 한 RNA → POC용 omics_cls 만 사용
            samples_full = [regions, x_rna]
            _, omics_cls_poc, _, _ = model(samples_full)

        k += 1
        # [수정] detach 없이 전부 grad 연결된 채로 누적
        image_feats_list.append(img_cls.squeeze(1))
        omics_feats_list.append(omics_cls_poc.squeeze(1))

        for b in range(img_cls.shape[0]):
            image_embeds.append(image_embed[b : b + 1])
            omics_embeds.append(omics_embed[b : b + 1])

        if k == accum_iter - 1:
            k = -1
            # [수정] torch.cat으로 합산 (모든 샘플 grad 연결)
            image_feats = torch.cat(image_feats_list, dim=0)
            omics_feats = torch.cat(omics_feats_list, dim=0)
            effective_batch = omics_feats.shape[0]

            # buffer 초기화
            image_feats_list = []
            omics_feats_list = []

            # ── 1. POC: Contrastive (omics 쪽은 마스킹 안 한 RNA 기준) ─────
            omics_feat = F.normalize(omics_feats, dim=1)
            image_feat = F.normalize(image_feats, dim=1)

            logit_scale = model.logit_scale.exp()
            sim_i2o     = logit_scale * image_feat @ omics_feat.t()
            sim_o2i     = sim_i2o.t()
            labels_poc  = torch.arange(effective_batch, device=device)

            loss_poc = (F.cross_entropy(sim_i2o, labels_poc) +
                        F.cross_entropy(sim_o2i, labels_poc)) / 2

            # POC 디버그 (epoch 0,1에서만)
            if epoch <= 1 and (data_iter_step + 1) % accum_iter == 0:
                logger.info(
                    f"[POC debug] loss_poc.requires_grad={loss_poc.requires_grad} "
                    f"image_feats.requires_grad={image_feats.requires_grad} "
                    f"omics_feats.requires_grad={omics_feats.requires_grad} "
                    f"logit_scale={model.logit_scale.exp().item():.4f}"
                )

            # ── 2. MOM ─────────────────────────────────────────────────────
            rna_mask_stack = torch.cat(rna_masks, dim=0)
            logits_mask_list, logits_cls_list = [], []

            for b in range(effective_batch):
                logit_mask, logit_cls = model.path_guided_omics_encoder(
                    image_embeds[b], omics_embeds[b])
                logits_mask_list.append(logit_mask)
                logits_cls_list.append(logit_cls)

            logits_mask_stack = torch.cat(logits_mask_list, dim=0)
            loss_mom = F.mse_loss(logits_mask_stack, rna_mask_stack)

            # ── 3. POM ───────────────────────────────────────────────────
            with torch.no_grad():
                weights_i2o = F.softmax(sim_i2o, dim=1).clone()
                weights_o2i = F.softmax(sim_o2i, dim=1).clone()
                weights_i2o.fill_diagonal_(0)
                weights_o2i.fill_diagonal_(0)

            try:
                for b in range(effective_batch):
                    neg_idx = torch.multinomial(weights_o2i[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(
                        image_embeds[neg_idx], omics_embeds[b])
                    logits_cls_list.append(logit_cls)

                for b in range(effective_batch):
                    neg_idx = torch.multinomial(weights_i2o[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(
                        image_embeds[b], omics_embeds[neg_idx])
                    logits_cls_list.append(logit_cls)

            except Exception as e:
                image_embeds, omics_embeds, rna_masks = [], [], []
                logger.info(f"POM negative sampling 오류: {e}")
                continue

            n_cls = len(logits_cls_list) // 3
            itm_labels = torch.cat([
                torch.ones(n_cls, dtype=torch.long),
                torch.zeros(2 * n_cls, dtype=torch.long)
            ], dim=0).to(device)

            logits_cls_stack = torch.cat(logits_cls_list, dim=0)
            loss_pom = F.cross_entropy(logits_cls_stack, itm_labels)

            image_embeds, omics_embeds, rna_masks = [], [], []

            # ── Total Loss ────────────────────────────────────────────────
            mom_scale  = 1.0 / math.sqrt(n_genes)
            mom_weight = getattr(args, "mom_weight", 0.3)
            loss = loss_poc * 1.0 + loss_pom * 6.0 + (loss_mom * mom_scale) * mom_weight

            metric_logger.update(loss_poc=loss_poc.item())
            metric_logger.update(loss_pom=loss_pom.item())
            metric_logger.update(loss_mom=loss_mom.item())

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.info(f"Loss={loss_value}, 학습 중단")
                sys.exit(1)

            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        clip_grad=max_norm if max_norm else None,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            logger.info(f"[{data_iter_step+1}] loss={loss_value:.4f}  "
                        f"poc={loss_poc.item():.4f}  "
                        f"pom={loss_pom.item():.4f}  "
                        f"mom={loss_mom.item():.4f}  "
                        f"(mom_scale=1/sqrt(n_genes), mom_weight={mom_weight})")

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer and (data_iter_step + 1) % accum_iter == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}, model