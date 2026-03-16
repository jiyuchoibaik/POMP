# main_multimodal_pretrain_ppi.py  ─ PPI 클러스터 RNA 사용 (rna_dim = n_clusters from pkl)
# pre-training/main_multimodal_pretrain_2.py 복제, rna_dim은 pkl의 n_genes(n_clusters) 사용.
import datetime
import os
import sys
import json
import pickle
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# ppi_rna 폴더 기준으로 import (실행 시 cwd가 ppi_rna 또는 프로젝트 루트일 수 있음)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.lr_decay as lrd

from utils.data_loader import build_dataset
from model.models_pomp import vit_base_patch16
from engine_multimodal_pretrain_2 import train_one_epoch

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    ap = argparse.ArgumentParser("TCGA-LUAD Multimodal Pretraining (PPI cluster RNA)")

    ap.add_argument("--data_pkl", default="./datasets/rna_ppi_clusters.pkl",
                    help="PPI 클러스터 전처리 결과 pkl (n_genes=n_clusters)")
    ap.add_argument("--n_genes", default=None, type=int,
                    help="미지정 시 pkl의 n_genes 사용 (n_clusters)")
    ap.add_argument("--max_patches", default=300,  type=int)

    ap.add_argument("--epochs",      default=501,  type=int)
    ap.add_argument("--batch_size",   default=1,    type=int)
    ap.add_argument("--accum_iter",   default=50,   type=int)
    ap.add_argument("--mask_ratio",   default=0.3,  type=float)
    ap.add_argument("--mom_weight",   default=0.3,  type=float)
    ap.add_argument("--num_workers",   default=8,    type=int)

    ap.add_argument("--lr",           default=5e-4, type=float)
    ap.add_argument("--min_lr",       default=1e-6, type=float)
    ap.add_argument("--warmup_epochs", default=50,  type=int)
    ap.add_argument("--weight_decay",  default=1e-5, type=float)
    ap.add_argument("--clip_grad",     default=1.0, type=float)
    ap.add_argument("--layer_decay",  default=0.75, type=float)

    ap.add_argument("--model",        default="vit_base_patch16", type=str)
    ap.add_argument("--drop_path",    default=0.1,  type=float)
    ap.add_argument("--global_pool",  action="store_true", default=True)

    ap.add_argument("--output_dir",   default="./output_pretrain_ppi")
    ap.add_argument("--log_dir",      default="./output_pretrain_ppi")
    ap.add_argument("--exptype",      default="tcga_luad_ppi")
    ap.add_argument("--save_every",   default=100,  type=int)
    ap.add_argument("--start_epoch",  default=0,    type=int)
    ap.add_argument("--resume",       default="",   type=str)

    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--seed",         default=200,  type=int)
    ap.add_argument("--pin_mem",      action="store_true", default=True)
    ap.add_argument("--dist_on_itp",   action="store_true", default=False)
    ap.add_argument("--distributed",   action="store_true", default=False)
    ap.add_argument("--world_size",   default=1,    type=int)
    ap.add_argument("--local_rank",   default=-1,    type=int)
    ap.add_argument("--dist_url",     default="env://")

    return ap.parse_args()


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    set_seed(args.seed)

    # PPI pkl에서 rna_dim(n_clusters) 읽기 (미지정 시)
    with open(args.data_pkl, "rb") as f:
        pkl_data = pickle.load(f)
    rna_dim = args.n_genes if args.n_genes is not None else int(pkl_data["n_genes"])
    logger.info(f"rna_dim (n_clusters) from pkl: {rna_dim}")

    dataset = build_dataset(args.data_pkl, max_num_region=args.max_patches)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = args.pin_mem,
        shuffle     = True,
        drop_last   = False,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir) if misc.get_rank() == 0 else None

    model = vit_base_patch16(
        rna_dim        = rna_dim,
        drop_path_rate = args.drop_path,
        global_pool    = args.global_pool,
        num_classes    = 0,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"파라미터 수: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    logger.info(f"학습 시작 (PPI cluster RNA): {args.epochs} epochs")
    start = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats, model = train_one_epoch(
            model, data_loader, optimizer,
            device, epoch, loss_scaler,
            max_norm   = args.clip_grad,
            log_writer = log_writer,
            args       = args,
        )

        if args.output_dir and epoch >= 50 and epoch % args.save_every == 0:
            misc.save_model(
                args=args, epoch=epoch, model=model,
                model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        logger.info(log_stats)

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            log_path = os.path.join(args.output_dir, f"log_pretrain_{args.exptype}.txt")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    logger.info(f"학습 완료. 소요시간: {elapsed}")


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
