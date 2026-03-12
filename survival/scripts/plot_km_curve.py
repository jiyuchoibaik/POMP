"""
5-fold out-of-fold 예측을 fold별 z-score 정규화한 뒤,
중위수로 고/저위험 그룹 나누고 Kaplan-Meier 곡선 그리기.
pvalue_full_dataset.py와 동일한 데이터·분할 사용.
"""
# Python 3.10 호환 (lifelines)
import datetime
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def load_zscore_data(result_dir):
    """predict_kfold_*.csv 로드 후 fold별 z-score, concat."""
    dfs = []
    for k in range(5):
        path = f"{result_dir}/predict_kfold_{k}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            break
        pred = df["predict"].values.astype(np.float64)
        mean, std = pred.mean(), pred.std()
        if std < 1e-9:
            std = 1.0
        df = df.copy()
        df["risk_z"] = (pred - mean) / std
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No predict_kfold_*.csv in {result_dir}")
    return pd.concat(dfs, axis=0, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="./output_finetune/predict_result_tcga_luad",
                    help="predict_kfold_0.csv ~ 4.csv 가 있는 디렉터리")
    ap.add_argument("--out", default="./output_finetune/km_curve.png", help="저장 경로")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    df = load_zscore_data(args.result_dir)
    n = len(df)
    pred_z = df["risk_z"].values
    censored = df["censored"].values.astype(bool)  # True = 사망 관측
    survival_days = df["survival"].values
    # 일 → 월 (365/12 ≈ 30.44일/월)
    survival_months = survival_days / (365.0 / 12.0)

    # 중위수로 고/저위험 (모델 해석: 낮을수록 고위험 → pred_z 낮으면 고위험)
    median_z = np.median(pred_z)
    high_risk = pred_z <= median_z  # 모델 기준 고위험
    low_risk = pred_z > median_z

    T_high = survival_months[high_risk]
    E_high = censored[high_risk]
    T_low = survival_months[low_risk]
    E_low = censored[low_risk]
    res = logrank_test(T_low, T_high, event_observed_A=E_low, event_observed_B=E_high)
    p_value = res.p_value

    # Kaplan-Meier (오차범위 = 신뢰구간 표시)
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    kmf.fit(T_high, event_observed=E_high, label=f"High risk (n={high_risk.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2166ac")
    kmf.fit(T_low, event_observed=E_low, label=f"Low risk (n={low_risk.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#f4a582")

    ax.set_xlabel("Follow up time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("TCGA-LUAD")
    ax.text(0.03, 0.03, f"p-value = {p_value:.5f}", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
