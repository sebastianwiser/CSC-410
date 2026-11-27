#!/usr/bin/env python3
"""
Top-5 equal-weight baseline using the most important variables.

Variables:
1) p2_entry
2) p1_entry
3) rank_points_diff
4) rank_diff
5) age_diff

Categorical entry variables are target-encoded using training data only.
All five features are z-scored on training data and averaged with equal weight.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import simple_model as sm

TOP5_NUMERIC = [
    "rank_points_diff",
    "rank_diff",
    "age_diff",
]

TOP5_CATEGORICAL = [
    "p1_entry",
    "p2_entry",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Top-5 equal-weight baseline for tennis match outcomes."
    )
    parser.add_argument(
        "--data-glob",
        default="atp_matches_*.csv",
        help="Glob pattern for match CSV files.",
    )
    parser.add_argument("--train-year-start", type=int, default=2014)
    parser.add_argument("--train-year-end", type=int, default=2022)
    parser.add_argument("--test-year-start", type=int, default=2023)
    parser.add_argument("--test-year-end", type=int, default=2024)
    return parser.parse_args()


def target_encoding(series: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"cat": series, "y": y})
    return df.groupby("cat", observed=True)["y"].mean().to_dict()


def entry_score(series: pd.Series, mapping: dict, default_value: float) -> pd.Series:
    return series.map(mapping).fillna(default_value)


def main():
    args = parse_args()

    df = sm.load_matches(args.data_glob)
    data = sm.build_dataset(df)

    cfg = sm.Config(
        data_glob=args.data_glob,
        train_year_start=args.train_year_start,
        train_year_end=args.train_year_end,
        test_year_start=args.test_year_start,
        test_year_end=args.test_year_end,
        save_model=None,
        plots_dir=None,
    )

    X_train, X_test, y_train, y_test = sm.make_splits(data, cfg)

    X_train = X_train[TOP5_NUMERIC + TOP5_CATEGORICAL]
    X_test = X_test[TOP5_NUMERIC + TOP5_CATEGORICAL]

    global_mean = y_train.mean()
    p1_map = target_encoding(X_train["p1_entry"], y_train)
    p2_map = target_encoding(X_train["p2_entry"], y_train)

    train_feat = pd.DataFrame({
        "rank_points_diff": X_train["rank_points_diff"],
        "rank_diff": X_train["rank_diff"],
        "age_diff": X_train["age_diff"],
        "p1_entry": entry_score(X_train["p1_entry"], p1_map, global_mean),
        "p2_entry": entry_score(X_train["p2_entry"], p2_map, global_mean),
    })

    test_feat = pd.DataFrame({
        "rank_points_diff": X_test["rank_points_diff"],
        "rank_diff": X_test["rank_diff"],
        "age_diff": X_test["age_diff"],
        "p1_entry": entry_score(X_test["p1_entry"], p1_map, global_mean),
        "p2_entry": entry_score(X_test["p2_entry"], p2_map, global_mean),
    })

    means = train_feat.mean()
    stds = train_feat.std().replace(0, 1)

    z_test = (test_feat - means) / stds

    score = z_test.mean(axis=1)
    probs = 1.0 / (1.0 + np.exp(-score))
    preds = (score >= 0).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print("Top-5 equal-weight baseline")
    print("Variables: p2_entry, p1_entry, rank_points_diff, rank_diff, age_diff")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")
    print(f"Test Brier:    {brier:.4f}")


if __name__ == "__main__":
    main()
