#!/usr/bin/env python3
"""
Simple baseline model with serve data (first serve % and win % stats).

Approach:
- Use only pre-match features (rank, rank points, age, height, surface, etc.)
- Add serve stats per player from gemini_serve_*.csv
- Improve data quality with name normalization + missing flags
- Add age curve and serve interaction features
- Train a regularized logistic regression classifier
- Optionally calibrate probabilities and run rolling-year evaluation
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_COLS = [
    "p1_rank",
    "p2_rank",
    "p1_rank_points",
    "p2_rank_points",
    "rank_diff",
    "rank_diff_abs",
    "rank_points_diff",
    "rank_points_diff_abs",
    "rank_ratio",
    "rank_points_ratio",
    "p1_rank_log",
    "p2_rank_log",
    "rank_log_diff",
    "p1_rank_points_log",
    "p2_rank_points_log",
    "rank_points_log_diff",
    "age_diff",
    "age_diff_abs",
    "age_diff_sq",
    "p1_age",
    "p2_age",
    "p1_age_sq",
    "p2_age_sq",
    "ht_diff",
    "ht_diff_abs",
    "seed_diff",
    "draw_size",
    "p1_first_serve_pct",
    "p2_first_serve_pct",
    "p1_first_serve_win_pct",
    "p2_first_serve_win_pct",
    "p1_second_serve_win_pct",
    "p2_second_serve_win_pct",
    "first_serve_pct_diff",
    "first_serve_win_pct_diff",
    "second_serve_win_pct_diff",
    "first_serve_pct_ratio",
    "first_serve_win_pct_ratio",
    "second_serve_win_pct_ratio",
    "p1_serve_missing",
    "p2_serve_missing",
    "best_of_is_5",
    "is_grand_slam",
    "serve_diff_best_of_5",
    "serve_win_diff_best_of_5",
    "serve2_diff_best_of_5",
    "serve_diff_grand_slam",
    "serve_win_diff_grand_slam",
    "serve2_diff_grand_slam",
    "serve_rank_interaction",
    "serve_win_rank_interaction",
    "serve2_rank_interaction",
    "rank_age_interaction",
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
    serve_glob: str
    train_year_start: int
    train_year_end: int
    test_year_start: int
    test_year_end: int
    calibrate: bool
    rolling_eval: bool
    model_type: str
    perm_importance: bool
    save_model: str | None
    plots_dir: str | None


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(normalize_name)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    b_safe = b.replace(0, np.nan)
    return a / b_safe


def load_matches(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    dfs = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_serve_stats(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No serve files found for pattern: {pattern}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        dfs.append(df)

    serve = pd.concat(dfs, ignore_index=True)
    serve = serve[[
        "player_name",
        "first_serve_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
    ]].copy()

    serve["player_name"] = serve["player_name"].astype(str).str.strip()
    serve["name_key"] = normalize_name_series(serve["player_name"])
    serve = serve[serve["name_key"] != ""]
    serve["first_serve_pct"] = pd.to_numeric(serve["first_serve_pct"], errors="coerce")
    serve["first_serve_win_pct"] = pd.to_numeric(serve["first_serve_win_pct"], errors="coerce")
    serve["second_serve_win_pct"] = pd.to_numeric(serve["second_serve_win_pct"], errors="coerce")

    serve = (
        serve.groupby("name_key", as_index=False)
        .agg(
            first_serve_pct=("first_serve_pct", "median"),
            first_serve_win_pct=("first_serve_win_pct", "median"),
            second_serve_win_pct=("second_serve_win_pct", "median"),
        )
        .reset_index(drop=True)
    )
    return serve


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "tourney_date",
        "surface",
        "tourney_level",
        "draw_size",
        "best_of",
        "round",
        "winner_name",
        "winner_hand",
        "winner_ht",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "winner_seed",
        "winner_entry",
        "loser_name",
        "loser_hand",
        "loser_ht",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
        "loser_seed",
        "loser_entry",
    ]
    df = df[keep].copy()

    df = df.dropna(
        subset=[
            "surface",
            "tourney_level",
            "best_of",
            "round",
            "winner_name",
            "winner_hand",
            "winner_ht",
            "winner_age",
            "winner_rank",
            "winner_rank_points",
            "loser_name",
            "loser_hand",
            "loser_ht",
            "loser_age",
            "loser_rank",
            "loser_rank_points",
            "tourney_date",
        ]
    )

    w = pd.DataFrame(
        {
            "tourney_date": df["tourney_date"],
            "surface": df["surface"],
            "tourney_level": df["tourney_level"],
            "draw_size": df["draw_size"],
            "best_of": df["best_of"],
            "round": df["round"],
            "p1_name": df["winner_name"],
            "p2_name": df["loser_name"],
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
        }
    )

    l = pd.DataFrame(
        {
            "tourney_date": df["tourney_date"],
            "surface": df["surface"],
            "tourney_level": df["tourney_level"],
            "draw_size": df["draw_size"],
            "best_of": df["best_of"],
            "round": df["round"],
            "p1_name": df["loser_name"],
            "p2_name": df["winner_name"],
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
        }
    )

    data = pd.concat([w, l], ignore_index=True)

    data["rank_diff"] = data["p1_rank"] - data["p2_rank"]
    data["rank_points_diff"] = data["p1_rank_points"] - data["p2_rank_points"]
    data["seed_diff"] = data["p1_seed"] - data["p2_seed"]
    data["age_diff"] = data["p1_age"] - data["p2_age"]
    data["ht_diff"] = data["p1_ht"] - data["p2_ht"]
    data["age_diff_sq"] = data["age_diff"] ** 2
    data["p1_age_sq"] = data["p1_age"] ** 2
    data["p2_age_sq"] = data["p2_age"] ** 2
    data["best_of_is_5"] = (pd.to_numeric(data["best_of"], errors="coerce") == 5).astype(int)
    data["is_grand_slam"] = (data["tourney_level"] == "G").astype(int)
    data["rank_diff_abs"] = data["rank_diff"].abs()
    data["rank_points_diff_abs"] = data["rank_points_diff"].abs()
    data["age_diff_abs"] = data["age_diff"].abs()
    data["ht_diff_abs"] = data["ht_diff"].abs()
    data["rank_ratio"] = safe_divide(data["p1_rank"], data["p2_rank"])
    data["rank_points_ratio"] = safe_divide(data["p1_rank_points"], data["p2_rank_points"])
    data["p1_rank_log"] = np.log1p(pd.to_numeric(data["p1_rank"], errors="coerce"))
    data["p2_rank_log"] = np.log1p(pd.to_numeric(data["p2_rank"], errors="coerce"))
    data["rank_log_diff"] = data["p1_rank_log"] - data["p2_rank_log"]
    data["p1_rank_points_log"] = np.log1p(pd.to_numeric(data["p1_rank_points"], errors="coerce"))
    data["p2_rank_points_log"] = np.log1p(pd.to_numeric(data["p2_rank_points"], errors="coerce"))
    data["rank_points_log_diff"] = data["p1_rank_points_log"] - data["p2_rank_points_log"]
    data["rank_age_interaction"] = data["rank_points_diff"] * data["age_diff"]

    return data


def attach_serve_stats(data: pd.DataFrame, serve: pd.DataFrame) -> pd.DataFrame:
    serve_map = serve.set_index("name_key")

    data["p1_name_key"] = normalize_name_series(data["p1_name"])
    data["p2_name_key"] = normalize_name_series(data["p2_name"])

    data["p1_first_serve_pct"] = data["p1_name_key"].map(serve_map["first_serve_pct"])
    data["p1_first_serve_win_pct"] = data["p1_name_key"].map(serve_map["first_serve_win_pct"])
    data["p1_second_serve_win_pct"] = data["p1_name_key"].map(serve_map["second_serve_win_pct"])

    data["p2_first_serve_pct"] = data["p2_name_key"].map(serve_map["first_serve_pct"])
    data["p2_first_serve_win_pct"] = data["p2_name_key"].map(serve_map["first_serve_win_pct"])
    data["p2_second_serve_win_pct"] = data["p2_name_key"].map(serve_map["second_serve_win_pct"])

    data["p1_serve_missing"] = (
        data[["p1_first_serve_pct", "p1_first_serve_win_pct", "p1_second_serve_win_pct"]]
        .isna()
        .any(axis=1)
        .astype(int)
    )
    data["p2_serve_missing"] = (
        data[["p2_first_serve_pct", "p2_first_serve_win_pct", "p2_second_serve_win_pct"]]
        .isna()
        .any(axis=1)
        .astype(int)
    )

    data["first_serve_pct_diff"] = data["p1_first_serve_pct"] - data["p2_first_serve_pct"]
    data["first_serve_win_pct_diff"] = (
        data["p1_first_serve_win_pct"] - data["p2_first_serve_win_pct"]
    )
    data["second_serve_win_pct_diff"] = (
        data["p1_second_serve_win_pct"] - data["p2_second_serve_win_pct"]
    )
    data["first_serve_pct_ratio"] = safe_divide(
        data["p1_first_serve_pct"], data["p2_first_serve_pct"]
    )
    data["first_serve_win_pct_ratio"] = safe_divide(
        data["p1_first_serve_win_pct"], data["p2_first_serve_win_pct"]
    )
    data["second_serve_win_pct_ratio"] = safe_divide(
        data["p1_second_serve_win_pct"], data["p2_second_serve_win_pct"]
    )

    data["serve_diff_best_of_5"] = data["first_serve_pct_diff"] * data["best_of_is_5"]
    data["serve_win_diff_best_of_5"] = data["first_serve_win_pct_diff"] * data["best_of_is_5"]
    data["serve2_diff_best_of_5"] = data["second_serve_win_pct_diff"] * data["best_of_is_5"]

    data["serve_diff_grand_slam"] = data["first_serve_pct_diff"] * data["is_grand_slam"]
    data["serve_win_diff_grand_slam"] = data["first_serve_win_pct_diff"] * data["is_grand_slam"]
    data["serve2_diff_grand_slam"] = data["second_serve_win_pct_diff"] * data["is_grand_slam"]

    data["serve_rank_interaction"] = data["first_serve_pct_diff"] * data["rank_points_diff"]
    data["serve_win_rank_interaction"] = data["first_serve_win_pct_diff"] * data["rank_points_diff"]
    data["serve2_rank_interaction"] = data["second_serve_win_pct_diff"] * data["rank_points_diff"]

    return data


def year_from_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.slice(0, 4)
    return s.astype(int)


def make_splits(data: pd.DataFrame, cfg: Config):
    years = year_from_date(data["tourney_date"])
    train_mask = (years >= cfg.train_year_start) & (years <= cfg.train_year_end)
    test_mask = (years >= cfg.test_year_start) & (years <= cfg.test_year_end)

    train = data[train_mask]
    test = data[test_mask]

    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Check year ranges and available files.")

    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train["label"]
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]
    y_test = test["label"]
    return X_train, X_test, y_train, y_test


def make_onehot(dense: bool) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def build_preprocessor(dense: bool = True) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", make_onehot(dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )

def build_model(model_type: str) -> Pipeline:
    preprocessor = build_preprocessor(dense=True)
    if model_type == "gb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=2000,
        scoring="roc_auc",
        n_jobs=-1,
        penalty="l2",
        solver="lbfgs",
        l1_ratios=(0.0,),
        use_legacy_attributes=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def fit_models(X_train, y_train, model_type: str, calibrate: bool):
    base_model = build_model(model_type)
    base_model.fit(X_train, y_train)

    pred_model = base_model
    if calibrate:
        pred_model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=5,
        )
        pred_model.fit(X_train, y_train)

    return base_model, pred_model


def evaluate(model: Pipeline, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    return acc, auc, brier, probs


def rolling_year_eval(data: pd.DataFrame, cfg: Config):
    years = year_from_date(data["tourney_date"])
    rows = []

    for year in range(cfg.test_year_start, cfg.test_year_end + 1):
        train_mask = (years >= cfg.train_year_start) & (years < year)
        test_mask = years == year

        train = data[train_mask]
        test = data[test_mask]
        if train.empty or test.empty:
            continue

        X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
        y_train = train["label"]
        X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]
        y_test = test["label"]

        _, pred_model = fit_models(X_train, y_train, cfg.model_type, cfg.calibrate)
        acc, auc, brier, _ = evaluate(pred_model, X_test, y_test)
        rows.append((year, acc, auc, brier, len(test)))

    if not rows:
        print("Rolling evaluation: no valid yearly splits.")
        return

    print("Rolling evaluation (train up to year-1, test on year):")
    for year, acc, auc, brier, n in rows:
        print(f"  {year}: acc={acc:.4f} auc={auc:.4f} brier={brier:.4f} n={n}")

    df = pd.DataFrame(rows, columns=["year", "acc", "auc", "brier", "n"])
    print("Rolling evaluation averages:")
    print(f"  acc={df['acc'].mean():.4f} auc={df['auc'].mean():.4f} brier={df['brier'].mean():.4f}")


def compute_variable_importance_logreg(model: Pipeline) -> pd.DataFrame:
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

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "variable": base_vars,
            "coef": coefs,
        }
    )
    df["importance"] = df["coef"].abs()
    var_imp = (
        df.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    var_imp["metric"] = "sum_abs_coef"
    return var_imp


def compute_variable_importance_permutation(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    if len(X_test) > 20000:
        X_sample = X_test.sample(20000, random_state=42)
        y_sample = y_test.loc[X_sample.index]
    else:
        X_sample = X_test
        y_sample = y_test

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    var_imp = pd.DataFrame(
        {
            "variable": X_sample.columns,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)
    var_imp["metric"] = "perm_importance_auc"
    var_imp = var_imp.reset_index(drop=True)
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

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve_serve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=30, alpha=0.8, color="#1f77b4")
    plt.xlabel("Predicted win probability (player1)")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prob_hist_serve.png"), dpi=150)
    plt.close()

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
    plt.savefig(os.path.join(out_dir, "calibration_serve.png"), dpi=150)
    plt.close()

    # Variable importance plot + CSV
    try:
        if var_imp is None:
            return
        var_imp_plot = var_imp.sort_values("importance", ascending=True)
        plt.figure(figsize=(7, 5))
        plt.barh(var_imp_plot["variable"], var_imp_plot["importance"], color="#2ca02c")
        metric = var_imp["metric"].iloc[0] if "metric" in var_imp.columns else "importance"
        plt.xlabel(f"Importance ({metric})")
        plt.title("Variable Importance (Serve Model)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_serve.png"), dpi=150)
        plt.close()

        var_imp.to_csv(
            os.path.join(out_dir, "variable_importance_serve.csv"),
            index=False,
        )
    except Exception:
        pass


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a tennis match model with serve stats.")
    parser.add_argument(
        "--data-glob",
        default="atp_matches_*.csv",
        help="Glob pattern for match CSV files.",
    )
    parser.add_argument(
        "--serve-glob",
        default="gemini_serve_*.csv",
        help="Glob pattern for serve CSV files.",
    )
    parser.add_argument("--train-year-start", type=int, default=2014)
    parser.add_argument("--train-year-end", type=int, default=2022)
    parser.add_argument("--test-year-start", type=int, default=2023)
    parser.add_argument("--test-year-end", type=int, default=2024)
    parser.add_argument(
        "--no-calibrate",
        action="store_false",
        dest="calibrate",
        help="Disable probability calibration.",
    )
    parser.add_argument(
        "--no-rolling-eval",
        action="store_false",
        dest="rolling_eval",
        help="Disable rolling-year evaluation.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "gb"],
        default="gb",
        help="Model type: logreg or gb (gradient boosting).",
    )
    parser.add_argument(
        "--no-perm-importance",
        action="store_false",
        dest="perm_importance",
        help="Disable permutation importance (used for gb model).",
    )
    parser.add_argument("--save-model", default="model_serve.joblib")
    parser.add_argument("--plots-dir", default="plots")
    parser.set_defaults(calibrate=True, rolling_eval=True, perm_importance=True)
    args = parser.parse_args()

    return Config(
        data_glob=args.data_glob,
        serve_glob=args.serve_glob,
        train_year_start=args.train_year_start,
        train_year_end=args.train_year_end,
        test_year_start=args.test_year_start,
        test_year_end=args.test_year_end,
        calibrate=args.calibrate,
        rolling_eval=args.rolling_eval,
        model_type=args.model,
        perm_importance=args.perm_importance,
        save_model=args.save_model,
        plots_dir=args.plots_dir,
    )


def main():
    cfg = parse_args()

    df = load_matches(cfg.data_glob)
    serve = load_serve_stats(cfg.serve_glob)

    data = build_dataset(df)
    data = attach_serve_stats(data, serve)
    p1_missing = data["p1_serve_missing"].mean()
    p2_missing = data["p2_serve_missing"].mean()
    print(f"Serve stats missing rate: p1={p1_missing:.2%} p2={p2_missing:.2%}")
    print(f"Model type: {cfg.model_type} (calibrate={cfg.calibrate})")

    X_train, X_test, y_train, y_test = make_splits(data, cfg)

    base_model, pred_model = fit_models(X_train, y_train, cfg.model_type, cfg.calibrate)

    acc, auc, brier, probs = evaluate(pred_model, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")
    print(f"Test Brier:    {brier:.4f}")

    baseline_preds = (X_test["rank_diff"] < 0).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Baseline (better rank) accuracy: {baseline_acc:.4f}")

    var_imp = None
    if cfg.model_type == "logreg":
        var_imp = compute_variable_importance_logreg(base_model)
    elif cfg.perm_importance:
        var_imp = compute_variable_importance_permutation(pred_model, X_test, y_test)

    if var_imp is not None:
        metric = var_imp["metric"].iloc[0] if "metric" in var_imp.columns else "importance"
        print(f"Variable importance ({metric}):")
        for i, row in var_imp.iterrows():
            print(f"{i + 1:2d}. {row['variable']}: {row['importance']:.4f}")

    if cfg.save_model:
        joblib.dump(pred_model, cfg.save_model)
        print(f"Saved model to: {cfg.save_model}")

    if cfg.plots_dir:
        plot_outputs(pred_model, X_test, y_test, probs, cfg.plots_dir, var_imp=var_imp)
        print(f"Saved plots to: {cfg.plots_dir}/")

    if cfg.rolling_eval:
        rolling_year_eval(data, cfg)


if __name__ == "__main__":
    main()
