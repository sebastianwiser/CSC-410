#!/usr/bin/env python3
"""
Very simple baseline model for ATP match outcomes.

Approach:
- Use only pre-match features (rank, rank points, age, height, surface, etc.)
- Create two rows per match (winner as player1, loser as player1) to keep it symmetric
- Train a logistic regression classifier
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_COLS = [
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "ht_diff",
    "seed_diff",
    "draw_size",
]

CATEGORICAL_COLS = [
    "surface",
    "tourney_level",
    "round",
    "best_of",
    "p1_hand",
    "p2_hand",
    "p1_entry",
    "p2_entry",
]


@dataclass
class Config:
    data_glob: str
    train_year_start: int
    train_year_end: int
    test_year_start: int
    test_year_end: int
    save_model: str | None
    plots_dir: str | None


def load_matches(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    dfs = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns we need. Anything else risks leakage (post-match stats).
    keep = [
        "tourney_date",
        "surface",
        "tourney_level",
        "draw_size",
        "best_of",
        "round",
        "winner_hand",
        "winner_ht",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "winner_seed",
        "winner_entry",
        "loser_hand",
        "loser_ht",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
        "loser_seed",
        "loser_entry",
    ]
    df = df[keep].copy()

    # Drop rows with missing key values
    df = df.dropna(subset=[
        "surface",
        "tourney_level",
        "best_of",
        "round",
        "winner_hand",
        "winner_ht",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "loser_hand",
        "loser_ht",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
        "tourney_date",
    ])

    # Winner as player1
    w = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "draw_size": df["draw_size"],
        "best_of": df["best_of"],
        "round": df["round"],
        "p1_hand": df["winner_hand"],
        "p2_hand": df["loser_hand"],
        "p1_rank": df["winner_rank"],
        "p2_rank": df["loser_rank"],
        "p1_rank_points": df["winner_rank_points"],
        "p2_rank_points": df["loser_rank_points"],
        "p1_seed": df["winner_seed"],
        "p2_seed": df["loser_seed"],
        "p1_entry": df["winner_entry"],
        "p2_entry": df["loser_entry"],
        "p1_age": df["winner_age"],
        "p2_age": df["loser_age"],
        "p1_ht": df["winner_ht"],
        "p2_ht": df["loser_ht"],
        "label": 1,
    })

    # Loser as player1 (mirror row)
    l = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "draw_size": df["draw_size"],
        "best_of": df["best_of"],
        "round": df["round"],
        "p1_hand": df["loser_hand"],
        "p2_hand": df["winner_hand"],
        "p1_rank": df["loser_rank"],
        "p2_rank": df["winner_rank"],
        "p1_rank_points": df["loser_rank_points"],
        "p2_rank_points": df["winner_rank_points"],
        "p1_seed": df["loser_seed"],
        "p2_seed": df["winner_seed"],
        "p1_entry": df["loser_entry"],
        "p2_entry": df["winner_entry"],
        "p1_age": df["loser_age"],
        "p2_age": df["winner_age"],
        "p1_ht": df["loser_ht"],
        "p2_ht": df["winner_ht"],
        "label": 0,
    })

    data = pd.concat([w, l], ignore_index=True)

    # Feature engineering
    data["rank_diff"] = data["p1_rank"] - data["p2_rank"]
    data["rank_points_diff"] = data["p1_rank_points"] - data["p2_rank_points"]
    data["seed_diff"] = data["p1_seed"] - data["p2_seed"]
    data["age_diff"] = data["p1_age"] - data["p2_age"]
    data["ht_diff"] = data["p1_ht"] - data["p2_ht"]

    # Keep only model features and label
    model_cols = ["tourney_date"] + NUMERIC_COLS + CATEGORICAL_COLS + ["label"]
    data = data[model_cols]
    return data


def year_from_date(series: pd.Series) -> pd.Series:
    # tourney_date is yyyymmdd, stored as int or str
    s = series.astype(str).str.slice(0, 4)
    return s.astype(int)


def make_splits(data: pd.DataFrame, cfg: Config):
    years = year_from_date(data["tourney_date"])
    train_mask = (years >= cfg.train_year_start) & (years <= cfg.train_year_end)
    test_mask = (years >= cfg.test_year_start) & (years <= cfg.test_year_end)

    train = data[train_mask]
    test = data[test_mask]

    if train.empty or test.empty:
        raise ValueError(
            "Train/test split is empty. Check year ranges and available files."
        )

    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train["label"]
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]
    y_test = test["label"]
    return X_train, X_test, y_train, y_test


def build_model() -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def evaluate(model: Pipeline, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    return acc, auc, brier, probs


def compute_variable_importance(model: Pipeline) -> pd.DataFrame:
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    feature_names = pre.get_feature_names_out()
    coefs = clf.coef_.ravel()

    base_vars = []
    for name in feature_names:
        if name.startswith("num__"):
            base_vars.append(name[len("num__"):])
            continue
        if name.startswith("cat__"):
            raw = name[len("cat__"):]
            base = None
            for col in CATEGORICAL_COLS:
                prefix = f"{col}_"
                if raw.startswith(prefix):
                    base = col
                    break
            if base is None:
                base = raw.split("_", 1)[0]
            base_vars.append(base)
            continue
        base_vars.append(name)

    df = pd.DataFrame({
        "feature": feature_names,
        "variable": base_vars,
        "coef": coefs,
    })
    df["abs_coef"] = df["coef"].abs()
    var_imp = (
        df.groupby("variable", as_index=False)["abs_coef"]
        .sum()
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    return var_imp


def plot_outputs(
    model: Pipeline,
    X_test,
    y_test,
    probs,
    out_dir: str,
    var_imp: pd.DataFrame | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # Probability histogram
    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=30, alpha=0.8, color="#1f77b4")
    plt.xlabel("Predicted win probability (player1)")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prob_hist.png"), dpi=150)
    plt.close()

    # Calibration curve
    bins = 10
    bin_edges = [i / bins for i in range(bins + 1)]
    df = pd.DataFrame({"p": probs, "y": y_test.values})
    df["bin"] = pd.cut(df["p"], bin_edges, include_lowest=True)
    calib = df.groupby("bin", observed=True).agg(p_mean=("p", "mean"), y_mean=("y", "mean"))
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.plot(calib["p_mean"], calib["y_mean"], marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=150)
    plt.close()

    # Top coefficients (by absolute value)
    try:
        pre = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]
        feature_names = pre.get_feature_names_out()
        coefs = clf.coef_.ravel()
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coef_df["abs"] = coef_df["coef"].abs()
        top = coef_df.sort_values("abs", ascending=False).head(20)

        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["coef"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Coefficient")
        plt.title("Top 20 Coefficients (Absolute)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_coeffs.png"), dpi=150)
        plt.close()
    except Exception:
        # If feature names not available for any reason, skip coefficients plot.
        pass

    # Variable importance (sum of absolute coefficients per original feature)
    try:
        if var_imp is None:
            var_imp = compute_variable_importance(model)
        var_imp_plot = var_imp.sort_values("abs_coef", ascending=True)
        plt.figure(figsize=(7, 5))
        plt.barh(var_imp_plot["variable"], var_imp_plot["abs_coef"], color="#2ca02c")
        plt.xlabel("Importance (sum |coef|)")
        plt.title("Variable Importance (Logistic Regression)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance.png"), dpi=150)
        plt.close()

        var_imp.to_csv(
            os.path.join(out_dir, "variable_importance.csv"),
            index=False,
        )
    except Exception:
        # If anything goes wrong, skip variable importance plot/CSV.
        pass


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a simple tennis match model.")
    parser.add_argument(
        "--data-glob",
        default="atp_matches_*.csv",
        help="Glob pattern for match CSV files.",
    )
    parser.add_argument("--train-year-start", type=int, default=2014)
    parser.add_argument("--train-year-end", type=int, default=2022)
    parser.add_argument("--test-year-start", type=int, default=2023)
    parser.add_argument("--test-year-end", type=int, default=2024)
    parser.add_argument("--save-model", default="model.joblib")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()

    return Config(
        data_glob=args.data_glob,
        train_year_start=args.train_year_start,
        train_year_end=args.train_year_end,
        test_year_start=args.test_year_start,
        test_year_end=args.test_year_end,
        save_model=args.save_model,
        plots_dir=args.plots_dir,
    )


def main():
    cfg = parse_args()

    df = load_matches(cfg.data_glob)
    data = build_dataset(df)

    X_train, X_test, y_train, y_test = make_splits(data, cfg)

    model = build_model()
    model.fit(X_train, y_train)

    acc, auc, brier, probs = evaluate(model, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")
    print(f"Test Brier:    {brier:.4f}")
    baseline_preds = (X_test["rank_diff"] < 0).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Baseline (better rank) accuracy: {baseline_acc:.4f}")

    if cfg.save_model:
        joblib.dump(model, cfg.save_model)
        print(f"Saved model to: {cfg.save_model}")

    var_imp = compute_variable_importance(model)
    print("Variable importance (sum |coef|):")
    for i, row in var_imp.iterrows():
        print(f"{i + 1:2d}. {row['variable']}: {row['abs_coef']:.4f}")

    if cfg.plots_dir:
        plot_outputs(model, X_test, y_test, probs, cfg.plots_dir, var_imp=var_imp)
        print(f"Saved plots to: {cfg.plots_dir}/")


if __name__ == "__main__":
    main()
