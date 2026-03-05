import datetime
import numpy as np
import os
import time
import pickle
import random
import copy
from pathlib import Path
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))

import utils.lr_decay as lrd
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.data_loader import POMPDataset
from model import models_pomp
from engine_multimodal_survival import train_one_epoch, evaluate
from utils.options import get_args_parser_finetune, logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    data_cv_splits = pickle.load(open(args.data_dir, 'rb'))

    start_time   = time.time()
    best_predict = []

    for _k, data in data_cv_splits.items():

        logger.info("=" * 60)
        logger.info(f"5-fold Cross-validation ({_k+1}/{len(data_cv_splits)})")
        logger.info("=" * 60)

        set_seed(args.seed)

        dataset_train = POMPDataset(data=data, split="train")
        dataset_val   = POMPDataset(data=data, split="validation")
        dataset_test  = POMPDataset(data=data, split="test")

        if misc.get_rank() == 0 and args.log_dir and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, shuffle=True)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, shuffle=False)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, shuffle=False)

        # ── 모델 생성 ────────────────────────────────────────────────────
        # n_genes 인수 추가 (원본: 없음 → 수정: n_genes=args.n_genes)
        model = models_pomp.__dict__[args.model](
            num_classes   = args.nb_classes,
            drop_path_rate= args.drop_path,
            global_pool   = args.global_pool,
            n_genes       = args.n_genes,       # ← 추가
        )

        # ── Pre-trained 가중치 로드 ───────────────────────────────────────
        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            logger.info(f"Pre-trained 체크포인트 로드: {args.finetune}")

            ckpt_model = checkpoint.get('model', checkpoint)
            state_dict = model.state_dict()

            # 크기 불일치 키 제거 (linear: 300→n_genes 등)
            skip_keys = []
            for k in list(ckpt_model.keys()):
                if k in state_dict and ckpt_model[k].shape != state_dict[k].shape:
                    logger.info(f"  shape 불일치 스킵: {k} "
                                f"{ckpt_model[k].shape} → {state_dict[k].shape}")
                    skip_keys.append(k)
                    del ckpt_model[k]

            # pre-training에 없는 survival 전용 키 제거
            for k in ['risk_head.0.weight', 'risk_head.0.bias']:
                ckpt_model.pop(k, None)

            msg = model.load_state_dict(ckpt_model, strict=False)
            logger.info(f"  로드 결과: missing={msg.missing_keys[:5]}...")
            trunc_normal_(model.risk_head[0].weight, std=2e-5)

        model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"파라미터 수: {n_params/1e6:.2f}M")

        if args.lr is None:
            args.lr = args.blr
        optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        loss_scaler  = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model,
                        optimizer=optimizer, loss_scaler=loss_scaler)

        # ── 학습 루프 ─────────────────────────────────────────────────────
        max_c_index_val          = 0.0
        best_predict_split       = []
        list_for_early_stop      = []
        list_for_reload_best     = []
        best_model_path          = os.path.join(
            args.output_dir, f"best_model_{args.exptype}_fold{_k}.pth")

        for epoch in range(args.start_epoch, args.epochs):
            if len(list_for_early_stop) >= 5:
                logger.info("Early stopping")
                break

            if len(list_for_reload_best) >= 15:
                list_for_reload_best = []
                if os.path.exists(best_model_path):
                    logger.info("best model 재로드")
                    model.load_state_dict(torch.load(best_model_path))

            train_stats, model = train_one_epoch(
                model, data_loader_train, optimizer,
                device, epoch, loss_scaler,
                args.clip_grad, None,
                log_writer=log_writer, args=args)

            test_stats, pred, censored, survival = evaluate(
                data_loader_test, model, device)
            val_stats = test_stats

            logger.info(f"val  c-index: {val_stats['c-index']:.4f}")
            logger.info(f"test c-index: {test_stats['c-index']:.4f}")

            list_for_early_stop.append(val_stats['c-index'])
            list_for_reload_best.append(val_stats['c-index'])

            if val_stats['c-index'] > max_c_index_val:
                max_c_index_val      = val_stats['c-index']
                list_for_early_stop  = []
                list_for_reload_best = []

                best_predict_split = [
                    round(train_stats['c-index'], 4),
                    round(val_stats['c-index'],   4),
                    round(test_stats['c-index'],  4),
                    round(train_stats['p-value'], 10),
                    round(val_stats['p-value'],   10),
                    round(test_stats['p-value'],  10),
                ]
                logger.info(f"★ New best val c-index: {max_c_index_val:.4f}")

                # 결과 저장
                result_dir = os.path.join(
                    args.output_dir, f"predict_result_{args.exptype}")
                os.makedirs(result_dir, exist_ok=True)
                pd.DataFrame({
                    "predict":  pred.flatten(0).cpu().detach().numpy(),
                    "censored": censored.cpu().detach().numpy(),
                    "survival": survival.cpu().detach().numpy(),
                }).to_csv(os.path.join(result_dir, f"predict_kfold_{_k}.csv"),
                          index=False)

                torch.save(model.state_dict(), best_model_path)

            log_stats = {
                'epoch': epoch, 'kfold': _k,
                **{f'train_{k}': round(v, 8) for k, v in train_stats.items()},
                **{f'val_{k}':   round(v, 8) for k, v in val_stats.items()},
                **{f'test_{k}':  round(v, 8) for k, v in test_stats.items()},
            }
            logger.info(log_stats)

            if log_writer:
                log_writer.add_scalar('perf/val_cindex',  val_stats['c-index'],  epoch)
                log_writer.add_scalar('perf/test_cindex', test_stats['c-index'], epoch)

            if args.output_dir and misc.is_main_process() and log_writer:
                log_writer.flush()

        best_predict.append(best_predict_split)
        logger.info(f"fold {_k} best: {best_predict_split}\n")

    # ── 최종 결과 요약 ────────────────────────────────────────────────────
    bp = np.array(best_predict)
    logger.info(f"train c-index: {bp[:,0]}, mean={bp[:,0].mean():.4f}")
    logger.info(f"val   c-index: {bp[:,1]}, mean={bp[:,1].mean():.4f} ± {bp[:,1].std():.4f}")
    logger.info(f"test  c-index: {bp[:,2]}, mean={bp[:,2].mean():.4f} ± {bp[:,2].std():.4f}")

    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"총 학습 시간: {total}")

    # fold별 예측 결과 합치기
    df_all = pd.DataFrame()
    result_dir = os.path.join(args.output_dir, f"predict_result_{args.exptype}")
    for i in range(5):
        path = os.path.join(result_dir, f"predict_kfold_{i}.csv")
        if os.path.exists(path):
            df_all = pd.concat([df_all, pd.read_csv(path)], axis=0)
    df_all.to_csv(os.path.join(result_dir, "predict_result.csv"), index=False)


def get_args():
    import argparse
    ap = argparse.ArgumentParser("TCGA-LUAD Survival Fine-tuning")

    # 데이터
    ap.add_argument("--data_dir",    required=True,
                    help="luad_cv_splits.pkl 경로")
    ap.add_argument("--n_genes",     default=2000, type=int,
                    help="RNA-seq HVG 차원 (preprocess_rna.py와 동일하게)")

    # 모델
    ap.add_argument("--model",       default="vit_base_patch16")
    ap.add_argument("--finetune",    default="",
                    help="pre-training 체크포인트 경로 (없으면 scratch)")
    ap.add_argument("--nb_classes",  default=1, type=int)
    ap.add_argument("--drop_path",   default=0.1, type=float)
    ap.add_argument("--global_pool", action="store_true", default=True)

    # 학습
    ap.add_argument("--epochs",      default=100,  type=int)
    ap.add_argument("--batch_size",  default=1,    type=int)
    ap.add_argument("--accum_iter",  default=32,   type=int)
    ap.add_argument("--lr",          default=None, type=float)
    ap.add_argument("--blr",         default=1e-4, type=float)
    ap.add_argument("--weight_decay",default=0.01, type=float)
    ap.add_argument("--clip_grad",   default=1.0,  type=float)
    ap.add_argument("--warmup_epochs",default=5,   type=int)
    ap.add_argument("--min_lr",      default=1e-6, type=float)
    ap.add_argument("--layer_decay", default=0.75, type=float)

    ap.add_argument("--mixup",            default=0., type=float)
    ap.add_argument("--cutmix",           default=0., type=float)
    ap.add_argument("--cutmix_minmax",    default=None, nargs='+', type=float)
    ap.add_argument("--mixup_prob",       default=1.0, type=float)
    ap.add_argument("--mixup_switch_prob",default=0.5, type=float)
    ap.add_argument("--mixup_mode",       default='batch')
    ap.add_argument("--smoothing",        default=0.1, type=float)

    # 저장 / 기타
    ap.add_argument("--output_dir",  default="./output_finetune")
    ap.add_argument("--log_dir",     default="./output_finetune")
    ap.add_argument("--exptype",     default="tcga_luad")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--seed",        default=42, type=int)
    ap.add_argument("--num_workers", default=8,  type=int)
    ap.add_argument("--pin_mem",     action="store_true", default=True)
    ap.add_argument("--eval",        action="store_true", default=False)
    ap.add_argument("--start_epoch", default=0,  type=int)
    ap.add_argument("--resume",      default="")
    ap.add_argument("--dist_on_itp", action="store_true", default=False)
    ap.add_argument("--distributed", action="store_true", default=False)
    ap.add_argument("--world_size",  default=1,  type=int)
    ap.add_argument("--local_rank",  default=-1, type=int)
    ap.add_argument("--dist_url",    default="env://")

    return ap.parse_args()


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)