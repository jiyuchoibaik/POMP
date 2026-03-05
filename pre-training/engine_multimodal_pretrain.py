# engine_multimodal_pretrain.py  ─ TCGA-LUAD (RNA-seq 단일 omics)
import math
import sys
import random
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.options import logger


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
    batch_size = args.accum_iter
    mask_ratio = getattr(args, "mask_ratio", 0.3)   # gene masking 비율

    optimizer.zero_grad()
    if log_writer:
        print(f"log_dir: {log_writer.log_dir}")

    # accumulation buffer
    omics_feats  = None
    image_feats  = None
    image_embeds = []
    omics_embeds = []
    rna_masks    = []   # 마스킹된 원본 RNA 벡터 (MOM 학습용)
    k = -1

    for data_iter_step, (regions, x_rna) in enumerate(data_loader):

        if (data_iter_step + 1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        regions = regions.to(device, non_blocking=True)
        x_rna   = x_rna.to(device, non_blocking=True)   # (1, rna_dim)

        # ── Gene-level Masking (MOM: Masked Omics Modeling) ──────────────
        # 원본 RNA 저장 후 mask_ratio 비율로 0으로 마스킹
        rna_orig   = x_rna.clone()
        n_genes    = x_rna.shape[1]
        n_mask     = int(n_genes * mask_ratio)
        mask_idx   = torch.randperm(n_genes)[:n_mask]
        x_rna_masked        = x_rna.clone()
        x_rna_masked[:, mask_idx] = 0.0

        rna_masks.append(rna_orig[0])   # 원본 (재구성 타깃)

        # ── Forward ──────────────────────────────────────────────────────
        with torch.cuda.amp.autocast():
            samples = [regions, x_rna_masked]
            img_cls, omics_cls, image_embed, omics_embed = model(samples)

        k += 1
        if k == 0:
            omics_feats = omics_cls
            image_feats = img_cls
        else:
            omics_feats = torch.cat([omics_feats, omics_cls], dim=0)
            image_feats = torch.cat([image_feats, img_cls],   dim=0)
        image_embeds.append(image_embed)
        omics_embeds.append(omics_embed)

        # ── Loss 계산 (accum_iter 마다) ───────────────────────────────────
        if k == accum_iter - 1:
            k = -1

            # ── 1. POC: Pathology-Omics Contrastive Loss (CLIP 방식) ─────
            omics_feat = F.normalize(omics_feats, dim=1)
            image_feat = F.normalize(image_feats, dim=1)

            logit_scale = model.logit_scale.exp()
            sim_i2o     = logit_scale * image_feat @ omics_feat.t()
            sim_o2i     = sim_i2o.t()
            labels_poc  = torch.arange(batch_size, device=device)

            loss_poc = (F.cross_entropy(sim_i2o, labels_poc) +
                        F.cross_entropy(sim_o2i, labels_poc)) / 2

            # ── 2. MOM: Masked Omics Modeling Loss (MSE) ─────────────────
            rna_mask_stack = torch.stack(rna_masks, dim=0)  # (B, rna_dim)
            logits_mask, logits_cls = [], []

            for b in range(batch_size):
                logit_mask, logit_cls = model.path_guided_omics_encoder(
                    image_embeds[b], omics_embeds[b])
                logits_mask.append(logit_mask[0])
                logits_cls.append(logit_cls[0])

            logits_mask_stack = torch.stack(logits_mask, dim=0)  # (B, rna_dim)
            loss_mom = F.mse_loss(logits_mask_stack, rna_mask_stack)

            # ── 3. POM: Pathology-Omics Matching Loss (ITM 방식) ─────────
            with torch.no_grad():
                weights_i2o = F.softmax(sim_i2o, dim=1).clone()
                weights_o2i = F.softmax(sim_o2i, dim=1).clone()
                weights_i2o.fill_diagonal_(0)
                weights_o2i.fill_diagonal_(0)

            try:
                # negative omics for each image
                for b in range(batch_size):
                    neg_idx = torch.multinomial(weights_o2i[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(
                        image_embeds[neg_idx], omics_embeds[b])
                    logits_cls.append(logit_cls[0])

                # negative image for each omics
                for b in range(batch_size):
                    neg_idx = torch.multinomial(weights_i2o[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(
                        image_embeds[b], omics_embeds[neg_idx])
                    logits_cls.append(logit_cls[0])

            except Exception as e:
                image_embeds, omics_embeds, rna_masks = [], [], []
                logger.info(f"POM negative sampling 오류: {e}")
                continue

            # positive: batch_size개, negative: 2*batch_size개
            itm_labels = torch.cat([
                torch.ones(batch_size,     dtype=torch.long),
                torch.zeros(2*batch_size,  dtype=torch.long)
            ], dim=0).to(device)

            logits_cls_stack = torch.stack(logits_cls, dim=0)
            loss_pom = F.cross_entropy(logits_cls_stack, itm_labels)

            # buffer reset
            image_embeds, omics_embeds, rna_masks = [], [], []

            # ── Total Loss ────────────────────────────────────────────────
            loss = loss_poc * 1.0 + loss_pom * 6.0 + loss_mom * 3.0

            metric_logger.update(loss_poc=loss_poc.item())
            metric_logger.update(loss_pom=loss_pom.item() * 6.0)
            metric_logger.update(loss_mom=loss_mom.item() * 3.0)

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