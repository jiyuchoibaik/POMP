# main_pretrain.py  ─ TCGA-LUAD (RNA-seq 단일 omics)
import datetime
import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.lr_decay as lrd

from data_loader import build_dataset
from models_pomp import vit_base_patch16
from engine_multimodal_pretrain import train_one_epoch
from utils.options import logger


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


def get_args():
    import argparse
    ap = argparse.ArgumentParser("TCGA-LUAD Multimodal Pretraining")

    # 데이터
    ap.add_argument("--data_pkl",    default="./datasets/rna_processed.pkl",
                    help="preprocess_rna.py 출력 pickle")
    ap.add_argument("--n_genes",     default=2000, type=int,
                    help="RNA-seq HVG 차원 (preprocess_rna.py와 동일하게)")
    ap.add_argument("--max_patches", default=300,  type=int)

    # 학습
    ap.add_argument("--epochs",      default=501,  type=int)
    ap.add_argument("--batch_size",  default=1,    type=int)
    ap.add_argument("--accum_iter",  default=50,   type=int,
                    help="gradient accumulation = effective batch size")
    ap.add_argument("--mask_ratio",  default=0.3,  type=float,
                    help="gene masking 비율 (MOM)")
    ap.add_argument("--num_workers", default=8,    type=int)

    # 옵티마이저
    ap.add_argument("--lr",          default=5e-4, type=float)
    ap.add_argument("--min_lr",      default=1e-6, type=float)
    ap.add_argument("--warmup_epochs", default=50, type=int)
    ap.add_argument("--weight_decay", default=1e-5, type=float)
    ap.add_argument("--clip_grad",   default=None, type=float)
    ap.add_argument("--layer_decay", default=0.75, type=float)

    # 모델
    ap.add_argument("--model",       default="vit_base_patch16", type=str)
    ap.add_argument("--drop_path",   default=0.1,  type=float)
    ap.add_argument("--global_pool", action="store_true", default=True)

    # 저장
    ap.add_argument("--output_dir",  default="./output_pretrain")
    ap.add_argument("--log_dir",     default="./output_pretrain")
    ap.add_argument("--exptype",     default="tcga_luad")
    ap.add_argument("--save_every",  default=100,  type=int,
                    help="N 에폭마다 체크포인트 저장")
    ap.add_argument("--start_epoch", default=0,    type=int)
    ap.add_argument("--resume",      default="",   type=str)

    # 기타
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        default=200,  type=int)
    ap.add_argument("--pin_mem",     action="store_true", default=True)
    ap.add_argument("--dist_on_itp", action="store_true", default=False)
    ap.add_argument("--distributed", action="store_true", default=False)
    ap.add_argument("--world_size",  default=1,    type=int)
    ap.add_argument("--local_rank",  default=-1,   type=int)
    ap.add_argument("--dist_url",    default="env://")

    return ap.parse_args()


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    set_seed(args.seed)
    logger.info(f"seed: {args.seed}")

    # ── 데이터셋 ────────────────────────────────────────────────────────────
    dataset = build_dataset(args.data_pkl, max_num_region=args.max_patches)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = args.pin_mem,
        shuffle     = True,
        drop_last   = False,
    )

    # ── TensorBoard ─────────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir) \
        if misc.get_rank() == 0 else None

    # ── 모델 ────────────────────────────────────────────────────────────────
    model = vit_base_patch16(
        rna_dim      = args.n_genes,
        drop_path_rate = args.drop_path,
        global_pool  = args.global_pool,
        num_classes  = 0,   # head 미사용
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"파라미터 수: {n_params/1e6:.2f}M")

    # ── 옵티마이저 ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    logger.info(f"학습 시작: {args.epochs} epochs")
    start = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats, model = train_one_epoch(
            model, data_loader, optimizer,
            device, epoch, loss_scaler,
            max_norm   = args.clip_grad,
            log_writer = log_writer,
            args       = args,
        )

        # 체크포인트 저장
        if args.output_dir and epoch >= 50 and epoch % args.save_every == 0:
            misc.save_model(
                args=args, epoch=epoch, model=model,
                model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch}
        logger.info(log_stats)

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            log_path = os.path.join(
                args.output_dir,
                f"log_pretrain_{args.exptype}.txt"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    logger.info(f"학습 완료. 소요시간: {elapsed}")


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)