#!/usr/bin/env python3
"""
model_1.py  —  Baseline Logistic Regression
==============================================
The simplest reasonable ATP match predictor.

Data:   data/initial/atp_matches_YYYY.csv  (2014-2024, ~30 000 matches)
Model:  Logistic Regression with standard scaling + one-hot encoding

Features (all pre-match, no lookahead):
  rank_diff       — ranking difference (p1 - p2)
  log_rank_diff   — log(1 + rank_p1) - log(1 + rank_p2)
  rank_ratio      — p1_rank / p2_rank
  pts_diff        — ranking points difference
  age_diff        — age difference
  h2h_rate        — head-to-head win rate (p1 vs p2, from prior meetings only)
  surface         — Hard / Clay / Grass / Carpet
  round           — R128 → F
  best_of         — 3 or 5 sets
  tourney_level   — G (Slam) / M (Masters) / A (ATP500) / D (Davis) etc.

Validation strategies:
  1. Expanding window  — train on all years ≤ N, test on year N+1
  2. Rolling 3-year   — train on years N-2..N, test on year N+1
  3. Single holdout   — train 2014-2020, test 2021-2024

Outputs (plots/ folder):
  feature_coefficients.png   — standardised logistic regression coefficients
  walk_forward.png           — accuracy per test year (expanding + rolling)
  accuracy_by_surface.png    — accuracy split by surface
  accuracy_by_round.png      — accuracy split by round
  rank_gap_accuracy.png      — accuracy by rank gap bucket
  roc_curve.png              — ROC curve (final holdout model)
  calibration.png            — calibration curve
  confusion_matrix.png       — confusion matrix

Usage:
  cd CSC410Pres/models/model_1
  python model_1.py
"""

from __future__ import annotations

import os
import glob
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.compose     import ColumnTransformer
from sklearn.impute      import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics     import (accuracy_score, brier_score_loss,
                                  confusion_matrix, roc_auc_score, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.pipeline    import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "..", "..", "data", "initial")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Feature columns ───────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "rank_diff", "log_rank_diff", "rank_ratio",
    "pts_diff", "age_diff", "h2h_rate",
]
CAT_COLS = ["surface", "round", "best_of", "tourney_level"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_raw(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv files found in {data_dir}")

    frames = []
    for fpath in files:
        df = pd.read_csv(fpath, low_memory=False)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # Parse tournament date  (format: YYYYMMDD int)
    raw["match_date"] = pd.to_datetime(
        raw["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    raw["year"] = raw["match_date"].dt.year

    # Numeric coercion
    for col in ["winner_rank", "loser_rank", "winner_rank_points",
                "loser_rank_points", "winner_age", "loser_age"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Drop rows missing core identifiers
    raw = raw.dropna(subset=["match_date", "winner_name", "loser_name",
                              "winner_rank", "loser_rank"]).reset_index(drop=True)
    raw = raw.sort_values("match_date").reset_index(drop=True)

    print(f"  Loaded {len(raw):,} matches  "
          f"({int(raw['year'].min())}–{int(raw['year'].max())})")
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HEAD-TO-HEAD (no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def compute_h2h(raw: pd.DataFrame) -> pd.Series:
    """
    For each match return the winner's pre-match H2H win rate vs this opponent.
    Uses only matches that occurred BEFORE the current one (data is date-sorted).
    """
    h2h: dict[tuple, int] = defaultdict(int)
    rates = []

    for row in raw.itertuples(index=False):
        w, l = row.winner_name, row.loser_name
        w_wins = h2h[(w, l)]
        l_wins = h2h[(l, w)]
        total  = w_wins + l_wins
        rates.append(w_wins / total if total > 0 else 0.5)
        h2h[(w, l)] += 1

    return pd.Series(rates, index=raw.index)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYMMETRIC DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Each match becomes two rows (p1=winner / p1=loser) so the model cannot
    learn a positional bias.  Label = 1 if p1 won.
    """
    raw = raw.copy()
    raw["h2h_w"] = compute_h2h(raw)

    # Normalise categoricals
    for col in ["surface", "round", "tourney_level"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip().str.lower()
    raw["best_of"] = raw["best_of"].astype(str).str.strip()

    rows = []
    for r in raw.itertuples(index=False):
        w_rank = r.winner_rank
        l_rank = r.loser_rank
        w_pts  = float(r.winner_rank_points) if not pd.isna(r.winner_rank_points) else 0.0
        l_pts  = float(r.loser_rank_points)  if not pd.isna(r.loser_rank_points)  else 0.0
        w_age  = r.winner_age
        l_age  = r.loser_age
        h2h_w  = r.h2h_w

        shared = dict(
            year         = r.year,
            surface      = getattr(r, "surface",       "unknown"),
            round        = getattr(r, "round",         "unknown"),
            best_of      = getattr(r, "best_of",       "3"),
            tourney_level= getattr(r, "tourney_level", "unknown"),
        )

        age_diff_wl = (w_age - l_age) if not (pd.isna(w_age) or pd.isna(l_age)) else np.nan

        # ── perspective: p1 = winner ──────────────────────────────────────
        rows.append({**shared,
            "rank_diff":     w_rank - l_rank,
            "log_rank_diff": np.log1p(w_rank) - np.log1p(l_rank),
            "rank_ratio":    w_rank / max(l_rank, 1),
            "pts_diff":      w_pts - l_pts,
            "age_diff":      age_diff_wl,
            "h2h_rate":      h2h_w,
            "label":         1,
        })
        # ── perspective: p1 = loser ───────────────────────────────────────
        rows.append({**shared,
            "rank_diff":     l_rank - w_rank,
            "log_rank_diff": np.log1p(l_rank) - np.log1p(w_rank),
            "rank_ratio":    l_rank / max(w_rank, 1),
            "pts_diff":      l_pts - w_pts,
            "age_diff":      -age_diff_wl if not pd.isna(age_diff_wl) else np.nan,
            "h2h_rate":      1.0 - h2h_w,
            "label":         0,
        })

    data = pd.DataFrame(rows)
    print(f"  Dataset: {len(data):,} rows  ({data['year'].min()}–{data['year'].max()})")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline() -> Pipeline:
    num_pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_eval(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return (
        accuracy_score(y_te, preds),
        roc_auc_score(y_te, probs),
        brier_score_loss(y_te, probs),
        probs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VALIDATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(data: pd.DataFrame):
    """
    For each test year from 2016 to 2024:
      - Expanding window:  train on all years < test_year
      - Rolling 3-year:    train on (test_year-3) to (test_year-1)
      - Rank baseline:     always predict lower-ranked player wins
    Returns a DataFrame with one row per test year.
    """
    years = sorted(data["year"].unique())
    rows  = []

    for test_yr in years:
        if test_yr < 2016:          # need at least 2 training years
            continue

        # Expanding window
        tr_exp  = data[data["year"] < test_yr]
        te      = data[data["year"] == test_yr]
        if len(tr_exp) < 100 or len(te) < 50:
            continue

        X_te = te[NUMERIC_COLS + CAT_COLS]
        y_te = te["label"]

        m_exp = make_pipeline()
        acc_exp, auc_exp, _, _ = fit_eval(
            m_exp, tr_exp[NUMERIC_COLS + CAT_COLS], tr_exp["label"],
            X_te, y_te
        )

        # Rolling 3-year window
        tr_roll = data[(data["year"] >= test_yr - 3) & (data["year"] < test_yr)]
        m_roll  = make_pipeline()
        acc_roll, auc_roll, _, _ = fit_eval(
            m_roll, tr_roll[NUMERIC_COLS + CAT_COLS], tr_roll["label"],
            X_te, y_te
        )

        # Rank baseline (lower rank = better player = predicted winner)
        rank_baseline = accuracy_score(y_te, (te["rank_diff"] < 0).astype(int))

        rows.append(dict(
            test_year   = test_yr,
            acc_expanding = acc_exp,
            acc_rolling   = acc_roll,
            auc_expanding = auc_exp,
            auc_rolling   = auc_roll,
            rank_baseline = rank_baseline,
            n_test        = len(te),
        ))
        print(f"    {test_yr}: expanding={acc_exp:.4f}  rolling={acc_roll:.4f}  "
              f"rank_base={rank_baseline:.4f}  (n={len(te):,})")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FEATURE IMPORTANCE (logistic coefficients)
# ══════════════════════════════════════════════════════════════════════════════

def extract_feature_names_and_coefs(model: Pipeline):
    pre   = model.named_steps["pre"]
    clf   = model.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_[0]
    return names, coefs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fname: str):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_feature_coefficients(model: Pipeline):
    names, coefs = extract_feature_names_and_coefs(model)
    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["abs"] = df["coef"].abs()
    df = df.sort_values("abs", ascending=True).tail(20)

    colors = ["tomato" if c > 0 else "steelblue" for c in df["coef"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["feature"], df["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (standardised)")
    ax.set_title("Most Important Variables — Baseline Logistic Regression\n"
                 "(positive = favours p1 winning, negative = favours p2)")
    legend = [
        mpatches.Patch(facecolor="tomato",    label="Positive effect on p1 win"),
        mpatches.Patch(facecolor="steelblue", label="Negative effect on p1 win"),
    ]
    ax.legend(handles=legend)
    _save("feature_coefficients.png")


def plot_walk_forward(wf: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Accuracy
    ax1.plot(wf["test_year"], wf["acc_expanding"], "o-", color="steelblue",
             lw=2, label="Expanding window (train all prior years)")
    ax1.plot(wf["test_year"], wf["acc_rolling"],   "s--", color="darkorange",
             lw=2, label="Rolling 3-year window")
    ax1.plot(wf["test_year"], wf["rank_baseline"],  "^:", color="gray",
             lw=1.5, label="Rank baseline (always pick lower-ranked)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Walk-Forward Validation — Accuracy per Test Year")
    ax1.legend()
    ax1.set_ylim(0.55, 0.80)
    ax1.grid(axis="y", alpha=0.3)

    # AUC
    ax2.plot(wf["test_year"], wf["auc_expanding"], "o-", color="steelblue", lw=2)
    ax2.plot(wf["test_year"], wf["auc_rolling"],   "s--", color="darkorange", lw=2)
    ax2.set_ylabel("ROC AUC")
    ax2.set_xlabel("Test Year")
    ax2.set_title("Walk-Forward Validation — ROC AUC per Test Year")
    ax2.set_ylim(0.60, 0.85)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate sample sizes
    for _, row in wf.iterrows():
        ax1.annotate(f"n={row.n_test:,}", xy=(row.test_year, row.acc_expanding - 0.015),
                     ha="center", fontsize=7, color="gray")

    _save("walk_forward.png")


def plot_accuracy_by_surface(data: pd.DataFrame, model: Pipeline):
    surfaces = [s for s in data["surface"].unique()
                if s not in ("unknown", "nan") and pd.notna(s)]
    surfaces = sorted(surfaces)

    accs, baselines, ns = [], [], []
    for surf in surfaces:
        sub = data[data["surface"] == surf]
        if len(sub) < 100:
            continue
        probs = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        acc   = accuracy_score(sub["label"], (probs >= 0.5).astype(int))
        base  = accuracy_score(sub["label"], (sub["rank_diff"] < 0).astype(int))
        accs.append(acc)
        baselines.append(base)
        ns.append(len(sub))

    # Filter to only surfaces that had data
    valid_surfaces = [s for s, n in zip(surfaces, ns) if n >= 100]
    accs       = [a for a, n in zip(accs, ns) if n >= 100]
    baselines  = [b for b, n in zip(baselines, ns) if n >= 100]
    ns         = [n for n in ns if n >= 100]

    x = np.arange(len(valid_surfaces))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs,      w, color="steelblue",  alpha=0.85, label="Logistic Regression")
    ax.bar(x + w/2, baselines, w, color="lightgray",  alpha=0.85, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.title()}\n(n={n:,})" for s, n in zip(valid_surfaces, ns)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 0.80)
    ax.set_title("Accuracy by Surface — Baseline vs Rank Baseline")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.legend()
    _save("accuracy_by_surface.png")


def plot_accuracy_by_round(data: pd.DataFrame, model: Pipeline):
    round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F",
                   "RR", "BR"]
    rounds_present = data["round"].unique()
    rounds = [r for r in round_order if r.lower() in [x.lower() for x in rounds_present]]
    # Also include any rounds not in the predefined order
    extra = [r for r in rounds_present
             if r.lower() not in [x.lower() for x in round_order]
             and r not in ("unknown", "nan") and pd.notna(r)]
    rounds += extra

    accs, ns = [], []
    valid_rounds = []
    for rnd in rounds:
        sub = data[data["round"].str.lower() == rnd.lower()]
        if len(sub) < 50:
            continue
        probs = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        acc   = accuracy_score(sub["label"], (probs >= 0.5).astype(int))
        accs.append(acc)
        ns.append(len(sub))
        valid_rounds.append(rnd)

    x = np.arange(len(valid_rounds))
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, accs, color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={n:,})" for r, n in zip(valid_rounds, ns)], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 0.85)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_title("Accuracy by Round — Early rounds are more predictable")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", fontsize=8)
    _save("accuracy_by_round.png")


def plot_rank_gap_accuracy(data: pd.DataFrame, model: Pipeline):
    """Shows how model accuracy varies by how big the rank gap is."""
    probs  = model.predict_proba(data[NUMERIC_COLS + CAT_COLS])[:, 1]
    df     = data[["rank_diff", "label"]].copy()
    df["prob"]    = probs
    df["correct"] = ((probs >= 0.5).astype(int) == df["label"]).astype(int)
    df["abs_gap"] = df["rank_diff"].abs()

    bins   = [0, 10, 25, 50, 100, 200, 500, 2000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
    df["gap_bin"] = pd.cut(df["abs_gap"], bins=bins, labels=labels)

    grp = df.groupby("gap_bin", observed=True).agg(
        acc=("correct", "mean"), n=("correct", "count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(grp["gap_bin"].astype(str), grp["acc"],
                  color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="Chance")
    ax.set_xlabel("Rank gap between players (|rank_p1 − rank_p2|)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Accuracy by Rank Gap\n"
                 "(larger rank gap = more predictable match)")
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{row['acc']:.3f}\n(n={int(row['n']):,})",
                ha="center", fontsize=8)
    _save("rank_gap_accuracy.png")


def plot_roc_curve(y_te, probs, acc, auc):
    fpr, tpr, _ = roc_curve(y_te, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="steelblue",
            label=f"Logistic Regression (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Baseline Model\n(Test 2021-2024 | Accuracy = {acc:.4f})")
    ax.legend()
    _save("roc_curve.png")


def plot_calibration(y_te, probs):
    fop, mpv = calibration_curve(y_te, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="steelblue", lw=2, label="Logistic Regression")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — How well do probabilities reflect reality?")
    ax.legend()
    _save("calibration.png")


def plot_confusion_matrix(y_te, probs):
    preds = (probs >= 0.5).astype(int)
    cm    = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Loss", "Predicted Win"], fontsize=11)
    ax.set_yticklabels(["Actual Loss", "Actual Win"],       fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=16, color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title("Confusion Matrix — Test Set (2021-2024)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save("confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  Model 1 — Baseline Logistic Regression")
    print("═" * 60)

    # ── Load & build dataset ──────────────────────────────────────────────────
    print("\nLoading data ...")
    raw  = load_raw(DATA_DIR)
    print("\nBuilding symmetric dataset ...")
    data = build_dataset(raw)

    # ── Walk-forward validation ───────────────────────────────────────────────
    print("\nRunning walk-forward validation ...")
    wf = walk_forward(data)

    # ── Final holdout model: train 2014-2020, test 2021-2024 ─────────────────
    print("\nTraining final model (2014-2020 train / 2021-2024 test) ...")
    train = data[data["year"] <= 2020]
    test  = data[data["year"] >= 2021]

    final_model = make_pipeline()
    acc, auc, brier, probs = fit_eval(
        final_model,
        train[NUMERIC_COLS + CAT_COLS], train["label"],
        test[NUMERIC_COLS + CAT_COLS],  test["label"],
    )
    rank_base = accuracy_score(test["label"], (test["rank_diff"] < 0).astype(int))

    print(f"\n{'═'*55}")
    print(f"{'Metric':<25} {'Value':>10}")
    print(f"{'─'*55}")
    print(f"{'Accuracy':<25} {acc:>10.4f}")
    print(f"{'Rank baseline accuracy':<25} {rank_base:>10.4f}")
    print(f"{'ROC AUC':<25} {auc:>10.4f}")
    print(f"{'Brier Score':<25} {brier:>10.4f}  (↓ better)")
    print(f"{'Train rows':<25} {len(train):>10,}")
    print(f"{'Test rows':<25} {len(test):>10,}")
    print(f"{'═'*55}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots ...")
    plot_feature_coefficients(final_model)
    plot_walk_forward(wf)
    plot_accuracy_by_surface(data, final_model)
    plot_accuracy_by_round(data, final_model)
    plot_rank_gap_accuracy(test, final_model)
    plot_roc_curve(test["label"], probs, acc, auc)
    plot_calibration(test["label"], probs)
    plot_confusion_matrix(test["label"], probs)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
