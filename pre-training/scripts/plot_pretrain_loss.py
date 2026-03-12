#!/usr/bin/env python3
"""
Pre-training task(POC, MOM, POM)별 loss 변화를 (a) Pre-training loss under different tasks 스타일로 시각화.
로그 파일: output_pretrain/log_pretrain_{exptype}.txt (JSON lines)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path: str):
    """log_pretrain_*.txt (한 줄당 JSON) 파싱 → epochs, total, poc, pom, mom 리스트."""
    epochs, total, poc, pom, mom = [], [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = d.get("epoch")
            if e is None:
                continue
            epochs.append(e)
            total.append(d.get("train_loss", np.nan))
            poc.append(d.get("train_loss_poc", np.nan))
            pom.append(d.get("train_loss_pom", np.nan))
            mom.append(d.get("train_loss_mom", np.nan))
    return (
        np.array(epochs),
        np.array(total, dtype=float),
        np.array(poc, dtype=float),
        np.array(pom, dtype=float),
        np.array(mom, dtype=float),
    )


def main():
    ap = argparse.ArgumentParser(description="Plot pre-training loss (Total, POC, POM, MOM)")
    ap.add_argument("--log", default=None,
                    help="로그 파일 경로 (기본: output_pretrain/log_pretrain_{exptype}.txt)")
    ap.add_argument("--exptype", default="tcga_luad", help="실험 타입 (--log 미지정 시 파일명에 사용)")
    ap.add_argument("--out", default=None,
                    help="저장할 그림 경로 (기본: 로그와 같은 디렉터리/pretrain_loss.png)")
    ap.add_argument("--output_dir", default="./output_pretrain",
                    help="--log 미지정 시 로그 디렉터리")
    args = ap.parse_args()

    if args.log is None:
        args.log = os.path.join(args.output_dir, f"log_pretrain_{args.exptype}.txt")
    if not os.path.isfile(args.log):
        raise SystemExit(f"로그 파일이 없습니다: {args.log}\n먼저 pre-training을 실행해 주세요.")

    epochs, total, poc, pom, mom = load_log(args.log)
    if len(epochs) == 0:
        raise SystemExit("유효한 로그 라인이 없습니다.")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(epochs, total, color="C0", label="Total loss", linewidth=1.2)   # blue
    ax.plot(epochs, poc,   color="black", label="POC loss", linewidth=1.0)
    ax.plot(epochs, pom,   color="#EDB120", label="POM loss", linewidth=1.0)  # yellow/gold
    ax.plot(epochs, mom,   color="C3", label="MOM loss", linewidth=1.0)     # red

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Pre-training loss under different tasks")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(args.log), "pretrain_loss.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
