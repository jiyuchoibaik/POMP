# engine_multimodal_pretrain.py  ─ TCGA-LUAD (RNA-seq 단일 omics)
import math
import sys
import random
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched
import logging; logger = logging.getLogger(__name__)

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
    header     = f"Epoch: [{epoch}]"
    print_freq = 1

    accum_iter = args.accum_iter
    mask_ratio = getattr(args, "mask_ratio", 0.3)

    optimizer.zero_grad()
    if log_writer:
        print(f"log_dir: {log_writer.log_dir}")

    # accumulation buffer
    omics_feats  = None
    image_feats  = None
    image_embeds = []
    omics_embeds = []
    rna_masks    = []
    k = -1

    for data_iter_step, (regions, x_rna) in enumerate(data_loader):

        if (data_iter_step + 1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        regions = regions.to(device, non_blocking=True)
        x_rna   = x_rna.to(device, non_blocking=True)   # (B, rna_dim)

        # ── Gene-level Masking ────────────────────────────────────────────
        rna_orig     = x_rna.clone()                     # (B, rna_dim)
        n_genes      = x_rna.shape[1]
        n_mask       = int(n_genes * mask_ratio)
        mask_idx     = torch.randperm(n_genes)[:n_mask]
        x_rna_masked = x_rna.clone()
        x_rna_masked[:, mask_idx] = 0.0

        rna_masks.append(rna_orig)   # (B, rna_dim)

        # ── Forward ──────────────────────────────────────────────────────
        with torch.cuda.amp.autocast():
            samples = [regions, x_rna_masked]
            img_cls, omics_cls, image_embed, omics_embed = model(samples)

        # img_cls, omics_cls: (B, 1, D) → squeeze to (B, D)
        # POC용 feature: 누적 시 detach하여 그래프 유지로 인한 gradient 불안정 방지.
        # 마지막 스텝만 grad 유지 (contrastive loss는 마지막 forward로만 역전파).
        k += 1
        is_last_accum = (k == accum_iter - 1)
        _img_cls = img_cls.squeeze(1)
        _omics_cls = omics_cls.squeeze(1)
        if not is_last_accum:
            _img_cls = _img_cls.detach()
            _omics_cls = _omics_cls.detach()
        if k == 0:
            omics_feats = _omics_cls
            image_feats = _img_cls
        else:
            omics_feats = torch.cat([omics_feats, _omics_cls], dim=0)
            image_feats = torch.cat([image_feats, _img_cls], dim=0)

        # image_embed, omics_embed를 배치별로 분리해서 저장
        for b in range(img_cls.shape[0]):
            image_embeds.append(image_embed[b:b+1])   # (1, seq, D)
            omics_embeds.append(omics_embed[b:b+1])   # (1, 2, D)

        # ── Loss 계산 (accum_iter 마다) ───────────────────────────────────
        if k == accum_iter - 1:
            k = -1

            effective_batch = omics_feats.shape[0]

            # ── 1. POC: Contrastive Loss ──────────────────────────────────
            omics_feat = F.normalize(omics_feats, dim=1)
            image_feat = F.normalize(image_feats, dim=1)

            logit_scale = model.logit_scale.exp()
            sim_i2o     = logit_scale * image_feat @ omics_feat.t()
            sim_o2i     = sim_i2o.t()
            labels_poc  = torch.arange(effective_batch, device=device)

            loss_poc = (F.cross_entropy(sim_i2o, labels_poc) +
                        F.cross_entropy(sim_o2i, labels_poc)) / 2

            # ── 2. MOM + POM positive: path_guided_omics_encoder 배치 한 번에 호출 ─
            rna_mask_stack = torch.cat(rna_masks, dim=0)  # (effective_batch, rna_dim)
            image_embeds_batch = torch.cat(image_embeds, dim=0)   # (effective_batch, seq, D)
            omics_embeds_batch = torch.cat(omics_embeds, dim=0)   # (effective_batch, 2, D)

            logits_mask_stack, logits_cls_pos = model.path_guided_omics_encoder(
                image_embeds_batch, omics_embeds_batch)
            # logits_mask_stack (effective_batch, rna_dim), logits_cls_pos (effective_batch, 2)
            loss_mom = F.mse_loss(logits_mask_stack, rna_mask_stack)

            # ── 3. POM: Pathology-Omics Matching (negative 샘플링도 배치로) ─────
            with torch.no_grad():
                weights_i2o = F.softmax(sim_i2o, dim=1).clone()
                weights_o2i = F.softmax(sim_o2i, dim=1).clone()
                weights_i2o.fill_diagonal_(0)
                weights_o2i.fill_diagonal_(0)

            try:
                neg_idx_o2i = torch.stack(
                    [torch.multinomial(weights_o2i[b], 1).squeeze(0) for b in range(effective_batch)])
                neg_idx_i2o = torch.stack(
                    [torch.multinomial(weights_i2o[b], 1).squeeze(0) for b in range(effective_batch)])

                image_neg_batch = torch.cat(
                    [image_embeds[neg_idx_o2i[b].item()] for b in range(effective_batch)], dim=0)
                _, logits_cls_neg1 = model.path_guided_omics_encoder(
                    image_neg_batch, omics_embeds_batch)

                omics_neg_batch = torch.cat(
                    [omics_embeds[neg_idx_i2o[b].item()] for b in range(effective_batch)], dim=0)
                _, logits_cls_neg2 = model.path_guided_omics_encoder(
                    image_embeds_batch, omics_neg_batch)
            except Exception as e:
                image_embeds, omics_embeds, rna_masks = [], [], []
                omics_feats = image_feats = None
                logger.info(f"POM negative sampling 오류: {e}")
                continue

            logits_cls_stack = torch.cat(
                [logits_cls_pos, logits_cls_neg1, logits_cls_neg2], dim=0)
            itm_labels = torch.cat([
                torch.ones(effective_batch,    dtype=torch.long),
                torch.zeros(2 * effective_batch, dtype=torch.long)
            ], dim=0).to(device)
            loss_pom = F.cross_entropy(logits_cls_stack, itm_labels)

            # buffer reset
            image_embeds, omics_embeds, rna_masks = [], [], []
            omics_feats = image_feats = None

            # ── Total Loss ────────────────────────────────────────────────
            loss = loss_poc * 1.0 + loss_pom * 6.0 + loss_mom * 3.0

            metric_logger.update(loss_poc=loss_poc.item())
            metric_logger.update(loss_pom=loss_pom.item())   # 원값 저장 (플롯용)
            metric_logger.update(loss_mom=loss_mom.item())

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.info(f"Loss={loss_value}, 학습 중단")
                sys.exit(1)

            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            logger.info(f"[{data_iter_step+1}] loss={loss_value:.4f}  "
                        f"poc={loss_poc.item():.4f}  "
                        f"pom={loss_pom.item():.4f}  "
                        f"mom={loss_mom.item():.4f}")

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer and (data_iter_step + 1) % accum_iter == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}, model