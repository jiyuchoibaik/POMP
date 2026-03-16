import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from tqdm import tqdm

import utils.misc as misc
import utils.lr_sched as lr_sched
from model.cox_loss import PartialLogLikelihood, calc_concordance_index, cox_log_rank
from utils.options import logger


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    outputs_accum  = None
    censored_accum = None
    survival_accum = None
    k = -1

    # ── 원본: (regions, X_mrna, X_mirna, X_meth, censored, survival)
    # ── 수정: (regions, X_rna, censored, survival)
    data_loader_tqdm = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Epoch {epoch}",
        unit="it",
        leave=False,
        disable=(misc.get_rank() != 0),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for data_iter_step, (regions, X_rna, censored, survival) in data_loader_tqdm:

        if (data_iter_step + 1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        regions  = regions.to(device,  non_blocking=True)
        X_rna    = X_rna.to(device,    non_blocking=True)
        censored = censored.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            samples = [regions, X_rna]
            outputs1, image_embed, omics_embed = model(samples)
            # risk1, path_guided risk 모두 (B,1) 반환 → (B,1) 유지 (unsqueeze 시 (B,1,1) 되어 Cox mm 오류)
            outputs2 = model.path_guided_omics_encoder(image_embed, omics_embed)
            outputs  = outputs1 + outputs2

        k += 1
        if k == 0:
            outputs_accum  = outputs
            censored_accum = censored
            survival_accum = survival
        else:
            outputs_accum  = torch.cat((outputs_accum,  outputs),  0)
            censored_accum = torch.cat((censored_accum, censored), 0)
            survival_accum = torch.cat((survival_accum, survival), 0)

        if k == accum_iter - 1 or data_iter_step == len(data_loader) - 1:
            k = -1

            # 생존시간 내림차순 정렬 (Cox loss 요구사항)
            order = torch.argsort(survival_accum, descending=True)
            outputs_accum  = outputs_accum[order]
            censored_accum = censored_accum[order]
            survival_accum = survival_accum[order]

            # concordance_index / cox_log_rank는 (N,) 필요. PartialLogLikelihood는 (N,1) 허용
            outputs_flat = outputs_accum.flatten()

            try:
                c_index = calc_concordance_index(outputs_flat, censored_accum, survival_accum)
                loss    = PartialLogLikelihood(outputs_accum, censored_accum, survival_accum)
                p_value = cox_log_rank(outputs_flat, censored_accum, survival_accum)
            except Exception as e:
                logger.info(f"loss 계산 오류: {e}")
                outputs_accum = censored_accum = survival_accum = None
                continue

            logger.info(f"train c-index: {c_index:.4f}, p-value: {p_value:.10f}")
            metric_logger.meters['c-index'].update(c_index, n=data_loader.batch_size)
            metric_logger.meters['p-value'].update(p_value, n=data_loader.batch_size)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.info(f"Loss={loss_value}, 학습 중단")
                sys.exit(1)

            loss /= accum_iter
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            metric_logger.update(loss=loss_value)
            lr = max(g["lr"] for g in optimizer.param_groups)
            metric_logger.update(lr=lr)

            if log_writer and (data_iter_step + 1) % accum_iter == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('loss', loss_value, epoch_1000x)
                log_writer.add_scalar('lr',   lr,         epoch_1000x)

    metric_logger.synchronize_between_processes()
    model.eval()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}, model


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    output_all   = torch.tensor([], device=device)
    censored_all = torch.tensor([], device=device)
    survival_all = torch.tensor([], device=device)

    data_loader_tqdm = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Eval",
        unit="it",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for regions, X_rna, censored, survival in data_loader_tqdm:
        regions  = regions.to(device,  non_blocking=True)
        X_rna    = X_rna.to(device,    non_blocking=True)
        censored = censored.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            samples  = [regions, X_rna]
            outputs1, image_embed, omics_embed = model(samples)
            outputs2 = model.path_guided_omics_encoder(image_embed, omics_embed)
            outputs  = outputs1 + outputs2

        output_all   = torch.cat((output_all,   outputs),  0)
        censored_all = torch.cat((censored_all, censored), 0)
        survival_all = torch.cat((survival_all, survival), 0)

    output_flat = output_all.flatten()
    c_index = calc_concordance_index(output_flat, censored_all, survival_all)
    p_value = cox_log_rank(output_flat, censored_all, survival_all)

    metric_logger.meters['c-index'].update(c_index, n=data_loader.batch_size)
    metric_logger.meters['p-value'].update(p_value, n=data_loader.batch_size)

    metric_logger.synchronize_between_processes()
    return ({k: m.global_avg for k, m in metric_logger.meters.items()},
            output_all, censored_all, survival_all)