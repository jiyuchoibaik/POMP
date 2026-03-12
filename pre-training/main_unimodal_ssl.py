# main_unimodal_ssl.py  ─ 이미지 단일 모달 SSL (slide-level contrastive)
# 멀티모달 vs 유니모달 WSI 집중 영역 비교를 위한 유니모달 인코더 학습.
# 동일 이미지 브랜치 구조, 학습 목표만 slide-level contrastive (두 뷰 → CLS 끌어당기기).
import os
import sys
import random
from pathlib import Path

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import build_dataset
from model.models_pomp import vit_base_patch16


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_branch_params(model):
    """이미지 브랜치 파라미터만 (유니모달 SSL에서는 이것만 학습)."""
    keys = ["patch_embed", "vits", "norm_vits", "cls_token", "img_transf", "norm_img_transf"]
    params = []
    for name, p in model.named_parameters():
        if any(name.startswith(k) for k in keys):
            params.append(p)
    return params


def train_one_epoch(model, loader, optimizer, device, N_sub, max_num_region, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        unit="batch",
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for regions, _ in pbar:
        regions = regions.to(device, non_blocking=True)
        B, N, C, H, W = regions.shape
        if N < N_sub * 2:
            continue
        idx = np.random.permutation(N)
        sub1 = idx[:N_sub]
        sub2 = idx[N_sub : N_sub * 2]
        view1 = regions[:, sub1].contiguous()
        view2 = regions[:, sub2].contiguous()

        cls1 = model.forward_image_only(view1)
        cls2 = model.forward_image_only(view2)
        cls1 = F.normalize(cls1, dim=1)
        cls2 = F.normalize(cls2, dim=1)

        logits = (cls1 @ cls2.T) / 0.07
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_batches, 1)


def main():
    import argparse
    ap = argparse.ArgumentParser("Unimodal SSL (image-only, slide-level contrastive)")
    ap.add_argument("--data_pkl", default="./datasets/rna_processed.pkl")
    ap.add_argument("--max_patches", default=250, type=int, help="샘플당 최대 region 수")
    ap.add_argument("--N_sub", default=125, type=int, help="한 뷰당 region 수 (두 뷰로 나눔)")
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--output_dir", default="./output_unimodal_ssl")
    ap.add_argument("--save_every", default=50, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--num_workers", default=16, type=int,
                    help="DataLoader 워커 수. GPU가 0%→100% 반복되면 로딩 병목이므로 16 등으로 늘리기")
    ap.add_argument("--resume", default="", help="체크포인트 경로 (이미지 브랜치만 로드 가능)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = build_dataset(args.data_pkl, max_num_region=args.max_patches)

    # pkl 경로 기준으로 wsi_paths 보정 (상대경로면 pkl이 있는 쪽 기준으로 해석)
    pkl_abspath = os.path.abspath(args.data_pkl)
    base_dir = os.path.dirname(os.path.dirname(pkl_abspath))
    new_paths = []
    for p in dataset.wsi_paths:
        if not p:
            new_paths.append(p)
            continue
        if not os.path.isabs(p):
            p = os.path.join(base_dir, os.path.normpath(p))
        new_paths.append(p)
    dataset.wsi_paths = new_paths

    # 실제 파일이 있는 케이스만 사용 (일부 케이스는 regions.npy 삭제된 경우 대비)
    valid_indices = [i for i in range(len(dataset)) if dataset.wsi_paths[i] and os.path.isfile(dataset.wsi_paths[i])]
    if len(valid_indices) < len(dataset):
        print(f"[Dataset] regions.npy 없는 케이스 제외: {len(dataset)} → {len(valid_indices)}개 사용")
    dataset = Subset(dataset, valid_indices)
    n_batches_per_epoch = len(dataset) // args.batch_size
    print(f"[준비] 유효 샘플 {len(dataset)}개, 에폭당 약 {n_batches_per_epoch} 스텝")

    loader_kw = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 8
        loader_kw["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[준비] 디바이스: {device}, 워커 {args.num_workers}개, 학습 시작...")
    model = vit_base_patch16(rna_dim=2000).to(device)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[Resume] loaded {args.resume} (strict=False)")

    params = image_branch_params(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    start_wall = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        loss = train_one_epoch(
            model, loader, optimizer, device, args.N_sub, args.max_patches,
            epoch + 1, args.epochs,
        )
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - start_wall
        remaining_epochs = args.epochs - (epoch + 1)
        eta_sec = epoch_elapsed * remaining_epochs if remaining_epochs > 0 else 0
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s" if eta_sec >= 0 else "-"
        print(f"Epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  (경과 {epoch_elapsed:.0f}s, 잔여 예상 약 {eta_str})")
        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            path = os.path.join(args.output_dir, f"checkpoint-unimodal-{epoch+1}.pth")
            torch.save(model.state_dict(), path)
            print(f"  saved {path}")

    path = os.path.join(args.output_dir, "checkpoint-unimodal-last.pth")
    torch.save(model.state_dict(), path)
    print(f"Done. Last checkpoint: {path}")


if __name__ == "__main__":
    main()
