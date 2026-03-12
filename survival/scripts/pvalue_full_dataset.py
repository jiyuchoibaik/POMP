"""
5-fold out-of-fold 예측을 fold별로 z-score 정규화한 뒤 합쳐서
전체 N명에 대해 C-index와 log-rank p-value 계산.

fold별 raw 예측 스케일이 다르므로, 각 fold test set 내에서 z-score로 바꾼 뒤
풀링하여 스케일을 맞춤.

사용법:
  python survival/scripts/pvalue_full_dataset.py \
    --result_dir ./output_finetune/predict_result_tcga_luad
"""
# Python 3.10 호환 (lifelines)
import datetime
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import argparse
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="./output_finetune/predict_result_tcga_luad",
                    help="predict_kfold_0.csv ~ 4.csv 가 있는 디렉터리")
    args = ap.parse_args()

    dfs = []
    for k in range(5):
        path = f"{args.result_dir}/predict_kfold_{k}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            break
        assert "predict" in df.columns and "censored" in df.columns and "survival" in df.columns
        pred = df["predict"].values.astype(np.float64)
        # fold 내 z-score (스케일 통일)
        mean, std = pred.mean(), pred.std()
        if std < 1e-9:
            std = 1.0
        df = df.copy()
        df["risk_z"] = (pred - mean) / std
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No predict_kfold_*.csv in {args.result_dir}")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    n = len(df_all)

    pred_z = df_all["risk_z"].values.astype(np.float64)
    censored = df_all["censored"].values.astype(np.int64)
    survival = df_all["survival"].values.astype(np.float64)

    # lifelines: higher predicted value = event sooner (worse). 모델이 높을수록 위험일 수도, 낮을수록 위험일 수도 있음.
    ci_pos = concordance_index(survival, pred_z, censored)   # pred_z 높을수록 위험 가정
    ci_neg = concordance_index(survival, -pred_z, censored)  # pred_z 낮을수록 위험 가정
    ci = max(ci_pos, ci_neg)
    if ci_neg > ci_pos:
        print(f"전체 샘플 수: {n} (fold별 z-score 정규화 후 풀링)")
        print(f"C-index (full {n}): {ci:.4f}  (예측 방향: 낮을수록 고위험으로 해석)")
    else:
        print(f"전체 샘플 수: {n} (fold별 z-score 정규화 후 풀링)")
        print(f"C-index (full {n}): {ci:.4f}  (예측 방향: 높을수록 고위험)")

    median = np.median(pred_z)
    low_risk = pred_z <= median
    high_risk = ~low_risk
    T_low = survival[low_risk]
    T_high = survival[high_risk]
    E_low = censored[low_risk]
    E_high = censored[high_risk]
    res = logrank_test(T_low, T_high, event_observed_A=E_low, event_observed_B=E_high)
    print(f"p-value (full {n}, log-rank): {res.p_value:.6f}")


if __name__ == "__main__":
    main()
