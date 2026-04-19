#!/usr/bin/env python3
"""
Model using all original variables plus play-style clusters from Gemini CSVs.

Original variables (same as simple_model.py):
- rank_diff, rank_points_diff, age_diff, ht_diff, seed_diff, draw_size
- surface, tourney_level, round, best_of, p1_hand, p2_hand, p1_entry, p2_entry

Play-style is clustered (KMeans) from:
- aggression_score
- court_depth_score

Style clusters are joined by player name from gemini_results_*.csv only.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
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
    "p1_style_cluster",
    "p2_style_cluster",
]


@dataclass
class Config:
    data_glob: str
    style_glob: str
    style_clusters: int
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


def load_styles(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No style files found for pattern: {pattern}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        dfs.append(df)

    styles = pd.concat(dfs, ignore_index=True)
    styles = styles[["player_name", "aggression_score", "court_depth_score"]].copy()
    styles["player_name"] = styles["player_name"].astype(str).str.strip()
    styles["name_key"] = styles["player_name"].str.lower()
    styles["aggression_score"] = pd.to_numeric(styles["aggression_score"], errors="coerce")
    styles["court_depth_score"] = pd.to_numeric(styles["court_depth_score"], errors="coerce")

    styles = (
        styles.groupby("name_key", as_index=False)
        .agg(
            aggression_score=("aggression_score", "mean"),
            court_depth_score=("court_depth_score", "mean"),
        )
        .reset_index(drop=True)
    )
    return styles


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
            "p1_entry": df["winner_entry"],
            "p2_entry": df["loser_entry"],
            "p1_rank": df["winner_rank"],
            "p2_rank": df["loser_rank"],
            "p1_rank_points": df["winner_rank_points"],
            "p2_rank_points": df["loser_rank_points"],
            "p1_seed": df["winner_seed"],
            "p2_seed": df["loser_seed"],
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
            "p1_entry": df["loser_entry"],
            "p2_entry": df["winner_entry"],
            "p1_rank": df["loser_rank"],
            "p2_rank": df["winner_rank"],
            "p1_rank_points": df["loser_rank_points"],
            "p2_rank_points": df["winner_rank_points"],
            "p1_seed": df["loser_seed"],
            "p2_seed": df["winner_seed"],
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

    return data


def build_style_cluster_map(styles: pd.DataFrame, n_clusters: int) -> dict:
    df = styles.dropna(subset=["aggression_score", "court_depth_score"]).copy()
    if df.empty:
        return {}

    n_clusters = max(1, min(int(n_clusters), len(df)))
    X = df[["aggression_score", "court_depth_score"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    df["cluster"] = labels
    return df.set_index("name_key")["cluster"].to_dict()


def attach_styles(data: pd.DataFrame, cluster_map: dict) -> pd.DataFrame:
    def label_cluster(x):
        if pd.isna(x):
            return "Unknown"
        return f"C{int(x)}"

    data["p1_name_key"] = data["p1_name"].astype(str).str.strip().str.lower()
    data["p2_name_key"] = data["p2_name"].astype(str).str.strip().str.lower()

    data["p1_style_cluster"] = data["p1_name_key"].map(cluster_map).map(label_cluster)
    data["p2_style_cluster"] = data["p2_name_key"].map(cluster_map).map(label_cluster)

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
        raise ValueError("Train/test split is empty. Check year ranges and files.")

    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train["label"]
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]
    y_test = test["label"]
    return X_train, X_test, y_train, y_test


def build_model() -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )
    clf = LogisticRegression(max_iter=1000)

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

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "variable": base_vars,
            "coef": coefs,
        }
    )
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

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve_style_top5.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=30, alpha=0.8, color="#1f77b4")
    plt.xlabel("Predicted win probability (player1)")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prob_hist_style_top5.png"), dpi=150)
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
    plt.savefig(os.path.join(out_dir, "calibration_style_top5.png"), dpi=150)
    plt.close()

    # Variable importance plot + CSV
    try:
        if var_imp is None:
            var_imp = compute_variable_importance(model)
        var_imp_plot = var_imp.sort_values("abs_coef", ascending=True)
        plt.figure(figsize=(7, 5))
        plt.barh(var_imp_plot["variable"], var_imp_plot["abs_coef"], color="#2ca02c")
        plt.xlabel("Importance (sum |coef|)")
        plt.title("Variable Importance (Style + Top 5)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variable_importance_style_top5.png"), dpi=150)
        plt.close()

        var_imp.to_csv(
            os.path.join(out_dir, "variable_importance_style_top5.csv"),
            index=False,
        )
    except Exception:
        pass


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Top-5 + play-style logistic regression model."
    )
    parser.add_argument(
        "--data-glob",
        default="atp_matches_*.csv",
        help="Glob pattern for match CSV files.",
    )
    parser.add_argument(
        "--style-glob",
        default="gemini_results_*.csv",
        help="Glob pattern for Gemini style CSV files.",
    )
    parser.add_argument(
        "--style-clusters",
        type=int,
        default=4,
        help="Number of play-style clusters to learn (default: 4).",
    )
    parser.add_argument("--train-year-start", type=int, default=2014)
    parser.add_argument("--train-year-end", type=int, default=2022)
    parser.add_argument("--test-year-start", type=int, default=2023)
    parser.add_argument("--test-year-end", type=int, default=2024)
    parser.add_argument("--save-model", default="model_top5_style.joblib")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()

    return Config(
        data_glob=args.data_glob,
        style_glob=args.style_glob,
        style_clusters=args.style_clusters,
        train_year_start=args.train_year_start,
        train_year_end=args.train_year_end,
        test_year_start=args.test_year_start,
        test_year_end=args.test_year_end,
        save_model=args.save_model,
        plots_dir=args.plots_dir,
    )


def main():
    cfg = parse_args()

    matches = load_matches(cfg.data_glob)
    styles = load_styles(cfg.style_glob)

    data = build_dataset(matches)
    cluster_map = build_style_cluster_map(styles, cfg.style_clusters)
    data = attach_styles(data, cluster_map)

    X_train, X_test, y_train, y_test = make_splits(data, cfg)

    model = build_model()
    model.fit(X_train, y_train)

    acc, auc, brier, probs = evaluate(model, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")
    print(f"Test Brier:    {brier:.4f}")

    var_imp = compute_variable_importance(model)
    print("Variable importance (sum |coef|):")
    for i, row in var_imp.iterrows():
        print(f"{i + 1:2d}. {row['variable']}: {row['abs_coef']:.4f}")

    if cfg.save_model:
        joblib.dump(model, cfg.save_model)
        print(f"Saved model to: {cfg.save_model}")

    if cfg.plots_dir:
        plot_outputs(model, X_test, y_test, probs, cfg.plots_dir, var_imp=var_imp)
        print(f"Saved plots to: {cfg.plots_dir}/")


if __name__ == "__main__":
    main()
