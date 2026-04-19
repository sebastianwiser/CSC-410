#!/usr/bin/env python3
"""
Advanced ATP match prediction model using atp_tennis.csv.

Key improvements over simple_model.py:
  - Larger dataset: ~67k matches from 2000-2026 (atp_tennis.csv)
  - XGBoost classifier instead of logistic regression
  - Rolling Elo ratings (overall + surface-specific) — updated after every match
  - Recent form (win rate over last 5 / 20 matches)
  - Surface-specific recent form
  - Head-to-head record between the two players
  - Days since last match (rest / fatigue proxy)
  - Match experience feature
  - Static features: Court (indoor/outdoor), Series level, rank ratio, log-rank diff
  - Optional bookmaker odds features (--use-odds)
  - Optional hyperparameter tuning via RandomizedSearchCV + TimeSeriesSplit (--tune)
  - Probability calibration via CalibratedClassifierCV (isotonic regression)
  - 9 diagnostic plots

Requirements: scikit-learn, xgboost, pandas, numpy, matplotlib, joblib
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Elo / form constants ────────────────────────────────────────────────────

ELO_BASE   = 1500.0
ELO_K      = 32.0   # overall Elo update step
ELO_K_SURF = 24.0   # surface-specific step (fewer matches per surface)
FORM_SHORT = 5      # recent-form window: 5 matches
FORM_LONG  = 20     # recent-form window: 20 matches

# ─── Feature column lists ────────────────────────────────────────────────────

STATIC_NUMERIC = [
    "rank_diff",
    "pts_diff",
    "rank_ratio",
    "log_rank_diff",
]

DYNAMIC_NUMERIC = [
    "elo_diff",          # overall Elo difference
    "elo_surf_diff",     # surface-specific Elo difference
    "form_short_diff",   # win-rate (last 5) difference
    "form_long_diff",    # win-rate (last 20) difference
    "surf_form_diff",    # surface win-rate difference
    "h2h_rate_p1",       # p1's H2H win rate vs p2 (0-1, 0.5 = no history)
    "h2h_n",             # total prior H2H matches
    "days_rest_diff",    # p1 days rest minus p2 days rest
    "experience_diff",   # p1 total matches minus p2 total matches
]

NUMERIC_COLS    = STATIC_NUMERIC + DYNAMIC_NUMERIC
CATEGORICAL_COLS = ["surface", "series", "court", "round", "best_of"]
ODDS_COLS        = ["odds_diff", "odds_ratio", "implied_prob_diff"]


@dataclass
class Config:
    data_path: str
    train_year_start: int
    train_year_end: int
    test_year_start: int
    test_year_end: int
    use_odds: bool
    tune: bool
    save_model: str | None
    plots_dir: str | None


# ─── Dynamic feature computation ─────────────────────────────────────────────

def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n: int | None = None) -> float:
    lst = list(dq)[-n:] if n else list(dq)
    return float(np.mean(lst)) if lst else 0.5


def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk matches chronologically and attach pre-match rolling stats to each row.
    Features for match i use only matches 0..i-1 — no look-ahead leakage.

    Added columns (prefix p1_ / p2_ relative to Player_1 / Player_2):
      p1_elo_pre, p2_elo_pre       — overall Elo
      p1_eloS_pre, p2_eloS_pre     — surface Elo
      p1_fS_pre, p2_fS_pre         — short-form (last 5 win rate)
      p1_fL_pre, p2_fL_pre         — long-form  (last 20 win rate)
      p1_sf_pre, p2_sf_pre         — surface-specific form
      p1_h2h_rate_pre              — p1's H2H win rate vs p2
      h2h_n_pre                    — total prior H2H matches
      p1_days_pre, p2_days_pre     — days since last match
      p1_exp_pre, p2_exp_pre       — career match count
    """
    df = df.sort_values("Date").reset_index(drop=True)

    elo_all   = defaultdict(lambda: ELO_BASE)
    elo_surf  = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all   = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf  = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_date = {}
    exp_cnt   = defaultdict(int)
    h2h_wins  = defaultdict(int)   # (winner_name, loser_name) → count

    feat_rows = []

    for row in df.itertuples(index=False):
        p1 = getattr(row, "Player_1", None)
        p2 = getattr(row, "Player_2", None)

        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win  = getattr(row, "Winner", None)
        surf = str(getattr(row, "Surface", "unknown") or "unknown").strip().lower()
        date = getattr(row, "Date", None)

        p1_won = (win == p1)

        # ── pre-match snapshots (read before any update) ──────────────────
        p1_elo  = elo_all[p1];    p2_elo  = elo_all[p2]
        p1_eloS = elo_surf[p1][surf]; p2_eloS = elo_surf[p2][surf]

        p1_fS = _form(res_all[p1], FORM_SHORT); p2_fS = _form(res_all[p2], FORM_SHORT)
        p1_fL = _form(res_all[p1]);             p2_fL = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf]);      p2_sf = _form(res_surf[p2][surf])

        h2h_p1    = h2h_wins[(p1, p2)]
        h2h_p2    = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        h2h_rate  = h2h_p1 / h2h_total if h2h_total > 0 else 0.5

        p1_days = (date - last_date[p1]).days if p1 in last_date else 30
        p2_days = (date - last_date[p2]).days if p2 in last_date else 30

        feat_rows.append({
            "p1_elo_pre":      p1_elo,
            "p2_elo_pre":      p2_elo,
            "p1_eloS_pre":     p1_eloS,
            "p2_eloS_pre":     p2_eloS,
            "p1_fS_pre":       p1_fS,
            "p2_fS_pre":       p2_fS,
            "p1_fL_pre":       p1_fL,
            "p2_fL_pre":       p2_fL,
            "p1_sf_pre":       p1_sf,
            "p2_sf_pre":       p2_sf,
            "p1_h2h_rate_pre": h2h_rate,
            "h2h_n_pre":       float(h2h_total),
            "p1_days_pre":     float(p1_days),
            "p2_days_pre":     float(p2_days),
            "p1_exp_pre":      float(exp_cnt[p1]),
            "p2_exp_pre":      float(exp_cnt[p2]),
        })

        # ── post-match updates ─────────────────────────────────────────────
        S  = 1.0 if p1_won else 0.0

        E  = _elo_expected(p1_elo, p2_elo)
        elo_all[p1] += ELO_K * (S - E)
        elo_all[p2] += ELO_K * ((1 - S) - (1 - E))

        ES = _elo_expected(p1_eloS, p2_eloS)
        elo_surf[p1][surf] += ELO_K_SURF * (S - ES)
        elo_surf[p2][surf] += ELO_K_SURF * ((1 - S) - (1 - ES))

        res_all[p1].append(1 if p1_won else 0)
        res_all[p2].append(0 if p1_won else 1)
        res_surf[p1][surf].append(1 if p1_won else 0)
        res_surf[p2][surf].append(0 if p1_won else 1)

        h2h_wins[(p1, p2) if p1_won else (p2, p1)] += 1

        if date is not None:
            last_date[p1] = date
            last_date[p2] = date
        exp_cnt[p1] += 1
        exp_cnt[p2] += 1

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ─── Data loading & feature engineering ─────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_dataset(df: pd.DataFrame, use_odds: bool = False) -> pd.DataFrame:
    df = df.rename(columns={"Best of": "best_of"}).copy()

    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["tourney_date"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    df["p1_wins"] = (df["Player_1"] == df["Winner"]).astype(int)
    df = df.dropna(subset=["Rank_1", "Rank_2", "Surface", "Round"]).reset_index(drop=True)

    for col in ["Surface", "Series", "Court", "Round"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
    df["best_of"] = df["best_of"].astype(str).str.strip()

    print("  Computing dynamic features (Elo, form, H2H) ...")
    df = compute_dynamic_features(df)

    rows_w = _make_rows(df, flip=False)
    rows_l = _make_rows(df, flip=True)
    data = pd.concat([rows_w, rows_l], ignore_index=True)

    # Static diff features
    data["rank_diff"]     = data["p1_rank"] - data["p2_rank"]
    data["pts_diff"]      = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"]    = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"] = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])

    # Dynamic diff features
    data["elo_diff"]        = data["p1_elo"]  - data["p2_elo"]
    data["elo_surf_diff"]   = data["p1_eloS"] - data["p2_eloS"]
    data["form_short_diff"] = data["p1_fS"]   - data["p2_fS"]
    data["form_long_diff"]  = data["p1_fL"]   - data["p2_fL"]
    data["surf_form_diff"]  = data["p1_sf"]   - data["p2_sf"]
    data["h2h_rate_p1"]     = data["h2h_rate"]
    data["h2h_n"]           = data["h2h_n_raw"]
    data["days_rest_diff"]  = data["p1_days"] - data["p2_days"]
    data["experience_diff"] = data["p1_exp"]  - data["p2_exp"]

    if use_odds:
        data["odds_diff"]         = data["p1_odds"] - data["p2_odds"]
        data["odds_ratio"]        = data["p1_odds"] / data["p2_odds"].clip(lower=0.01)
        data["implied_prob_diff"] = (
            1.0 / data["p1_odds"].clip(lower=0.01)
            - 1.0 / data["p2_odds"].clip(lower=0.01)
        )

    num_cols = NUMERIC_COLS + (ODDS_COLS if use_odds else [])
    keep = ["tourney_date"] + num_cols + CATEGORICAL_COLS + ["label"]
    return data[keep].copy()


def _make_rows(df: pd.DataFrame, flip: bool) -> pd.DataFrame:
    """Build one side of the symmetric dataset."""

    def _v(col_a: str, col_b: str) -> np.ndarray:
        col = col_b if flip else col_a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    label = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values

    # H2H rate is already from Player_1's perspective; when we flip, it becomes 1 - rate
    h2h_rate = (
        (1.0 - df["p1_h2h_rate_pre"].values)
        if flip else
        df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    return pd.DataFrame({
        "tourney_date": df["tourney_date"].values,
        # static
        "p1_rank":  _v("Rank_1", "Rank_2"),
        "p2_rank":  _v("Rank_2", "Rank_1"),
        "p1_pts":   _v("Pts_1",  "Pts_2"),
        "p2_pts":   _v("Pts_2",  "Pts_1"),
        "p1_odds":  _v("Odd_1",  "Odd_2"),
        "p2_odds":  _v("Odd_2",  "Odd_1"),
        # dynamic — swap p1/p2 sides when flipped
        "p1_elo":   _v("p1_elo_pre",  "p2_elo_pre"),
        "p2_elo":   _v("p2_elo_pre",  "p1_elo_pre"),
        "p1_eloS":  _v("p1_eloS_pre", "p2_eloS_pre"),
        "p2_eloS":  _v("p2_eloS_pre", "p1_eloS_pre"),
        "p1_fS":    _v("p1_fS_pre",   "p2_fS_pre"),
        "p2_fS":    _v("p2_fS_pre",   "p1_fS_pre"),
        "p1_fL":    _v("p1_fL_pre",   "p2_fL_pre"),
        "p2_fL":    _v("p2_fL_pre",   "p1_fL_pre"),
        "p1_sf":    _v("p1_sf_pre",   "p2_sf_pre"),
        "p2_sf":    _v("p2_sf_pre",   "p1_sf_pre"),
        "h2h_rate": h2h_rate,
        "h2h_n_raw":df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
        "p1_days":  _v("p1_days_pre", "p2_days_pre"),
        "p2_days":  _v("p2_days_pre", "p1_days_pre"),
        "p1_exp":   _v("p1_exp_pre",  "p2_exp_pre"),
        "p2_exp":   _v("p2_exp_pre",  "p1_exp_pre"),
        # categorical
        "surface":  df["Surface"].values,
        "series":   df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
        "court":    df["Court"].values  if "Court"  in df.columns else np.full(len(df), "unknown"),
        "round":    df["Round"].values,
        "best_of":  df["best_of"].values,
        "label":    label,
    })


def year_from_date(series: pd.Series) -> pd.Series:
    return series.astype(str).str[:4].astype(int)


def make_splits(data: pd.DataFrame, cfg: Config, use_odds: bool = False):
    years    = year_from_date(data["tourney_date"])
    tr_mask  = (years >= cfg.train_year_start) & (years <= cfg.train_year_end)
    te_mask  = (years >= cfg.test_year_start)  & (years <= cfg.test_year_end)

    train, test = data[tr_mask], data[te_mask]
    if train.empty or test.empty:
        raise ValueError("Empty train/test split — check year ranges.")

    feat_cols = NUMERIC_COLS + (ODDS_COLS if use_odds else []) + CATEGORICAL_COLS
    X_tr = train[feat_cols]; y_tr = train["label"]
    X_te = test[feat_cols];  y_te = test["label"]
    meta = test[["tourney_date", "surface", "series", "round"]].copy()
    meta.index = y_te.index
    return X_tr, X_te, y_tr, y_te, meta


# ─── Model ───────────────────────────────────────────────────────────────────

def _make_preprocessor(use_odds: bool) -> ColumnTransformer:
    num_cols = NUMERIC_COLS + (ODDS_COLS if use_odds else [])
    num_pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])


def build_and_fit_model(X_tr, y_tr, use_odds: bool, tune: bool):
    preprocessor = _make_preprocessor(use_odds)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", xgb)])

    if tune:
        param_dist = {
            "clf__n_estimators":     [200, 400, 600],
            "clf__max_depth":        [3, 4, 5],
            "clf__learning_rate":    [0.01, 0.05, 0.1],
            "clf__subsample":        [0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.7, 0.8, 0.9],
            "clf__min_child_weight": [10, 20, 30],
        }
        tscv   = TimeSeriesSplit(n_splits=4)
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=20, cv=tscv,
            scoring="roc_auc", n_jobs=-1, random_state=42, verbose=1,
        )
        search.fit(X_tr, y_tr)
        print(f"Best CV AUC : {search.best_score_:.4f}")
        print(f"Best params : {search.best_params_}")
        # Rebuild with best params and calibrate properly (cv="prefit" removed in sklearn 1.2+)
        best_clf_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        xgb_best  = XGBClassifier(**{**xgb.get_params(), **best_clf_params})
        best_pipe = Pipeline([("pre", preprocessor), ("clf", xgb_best)])
        cal = CalibratedClassifierCV(best_pipe, cv=3, method="isotonic")
        cal.fit(X_tr, y_tr)
        return cal

    cal = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
    cal.fit(X_tr, y_tr)
    return cal


def evaluate(model, X_te, y_te):
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_te, preds)
    auc   = roc_auc_score(y_te, probs)
    brier = brier_score_loss(y_te, probs)
    return acc, auc, brier, probs


# ─── Plotting ────────────────────────────────────────────────────────────────

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc(y_te, probs, out_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_te, probs)
    auc = roc_auc_score(y_te, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    _savefig(os.path.join(out_dir, "roc_curve.png"))


def plot_calibration(y_te, probs, out_dir: str) -> None:
    fop, mpv = calibration_curve(y_te, probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(mpv, fop, "o-", label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    _savefig(os.path.join(out_dir, "calibration.png"))


def plot_prob_histogram(y_te, probs, out_dir: str) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    plt.figure(figsize=(8, 5))
    plt.hist(df[df.label == 1]["prob"], bins=30, alpha=0.6,
             label="Wins (label=1)", color="steelblue")
    plt.hist(df[df.label == 0]["prob"], bins=30, alpha=0.6,
             label="Losses (label=0)", color="tomato")
    plt.axvline(0.5, color="black", linestyle="--")
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution by Outcome")
    plt.legend()
    _savefig(os.path.join(out_dir, "prob_hist.png"))


def plot_confusion_matrix(y_te, probs, out_dir: str) -> None:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Loss", "Pred Win"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Loss", "True Win"])
    ax.set_title("Confusion Matrix")
    _savefig(os.path.join(out_dir, "confusion_matrix.png"))


def plot_precision_recall(y_te, probs, out_dir: str) -> None:
    prec, rec, _ = precision_recall_curve(y_te, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, color="darkorange")
    plt.axhline(0.5, color="gray", linestyle="--", label="Baseline (random)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    _savefig(os.path.join(out_dir, "precision_recall.png"))


def _extract_importances(model) -> tuple[list, np.ndarray] | None:
    try:
        if hasattr(model, "calibrated_classifiers_"):
            pipes = [cc.estimator for cc in model.calibrated_classifiers_]
        elif hasattr(model, "estimator"):
            pipes = [model.estimator]
        else:
            pipes = [model]

        imps = [p.named_steps["clf"].feature_importances_
                for p in pipes if hasattr(p, "named_steps")]
        if not imps:
            return None

        avg_imp = np.mean(imps, axis=0)
        names   = list(pipes[0].named_steps["pre"].get_feature_names_out())
        return names, avg_imp
    except Exception as e:
        print(f"Warning: feature importance extraction failed: {e}")
        return None


def plot_feature_importance(model, out_dir: str) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    top = df.head(25).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"], color="steelblue")
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Top 25 Feature Importances (XGBoost)")
    _savefig(os.path.join(out_dir, "feature_importance.png"))


def plot_accuracy_by_category(y_te, probs, meta, out_dir: str) -> None:
    df = meta.copy()
    df["pred"]    = (probs >= 0.5).astype(int)
    df["label"]   = y_te.values
    df["correct"] = (df["pred"] == df["label"]).astype(int)

    cats = [
        ("surface", "Accuracy by Surface"),
        ("series",  "Accuracy by Tournament Series"),
        ("round",   "Accuracy by Round"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, title) in zip(axes, cats):
        grp = (
            df.groupby(col)["correct"]
            .agg(accuracy="mean", count="count")
            .reset_index()
            .query("count >= 30")
            .sort_values("accuracy", ascending=True)
        )
        bars = ax.barh(grp[col], grp["accuracy"], color="steelblue")
        ax.axvline(0.5, color="gray", linestyle="--")
        ax.set_xlabel("Accuracy")
        ax.set_title(title)
        for bar, cnt in zip(bars, grp["count"]):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={cnt}", va="center", fontsize=8)
    _savefig(os.path.join(out_dir, "accuracy_by_category.png"))


def plot_accuracy_by_confidence(y_te, probs, out_dir: str) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    df["conf"]    = np.abs(df["prob"] - 0.5)
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)

    bins   = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
              for i in range(len(bins) - 1)]
    df["conf_bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
    grp = (df.groupby("conf_bin", observed=True)["correct"]
             .agg(accuracy="mean", count="count")
             .reset_index())

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(grp["conf_bin"], grp["accuracy"], alpha=0.7, color="steelblue", label="Accuracy")
    ax2.plot(grp["conf_bin"], grp["count"], "o-", color="tomato", label="Count")
    ax1.axhline(0.5, color="gray", linestyle="--")
    ax1.set_xlabel("Confidence Bucket (|P − 0.5|)")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Sample Count")
    ax1.set_title("Accuracy vs Prediction Confidence")
    plt.xticks(rotation=30, ha="right")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    _savefig(os.path.join(out_dir, "accuracy_by_confidence.png"))


def plot_yearly_accuracy(data: pd.DataFrame, model, cfg: Config,
                          use_odds: bool, out_dir: str) -> None:
    feat_cols = NUMERIC_COLS + (ODDS_COLS if use_odds else []) + CATEGORICAL_COLS
    years = year_from_date(data["tourney_date"])
    rows  = []
    for yr in sorted(years.unique()):
        mask = years == yr
        X_yr = data[mask][feat_cols]
        y_yr = data[mask]["label"]
        if len(y_yr) < 50:
            continue
        prb  = model.predict_proba(X_yr)[:, 1]
        acc  = accuracy_score(y_yr, (prb >= 0.5).astype(int))
        base = accuracy_score(y_yr, (X_yr["rank_diff"] < 0).astype(int))
        rows.append({
            "year":     yr,
            "accuracy": acc,
            "baseline": base,
            "split":    "test" if yr >= cfg.test_year_start else "train",
        })

    res = pd.DataFrame(rows)
    tr  = res[res.split == "train"]
    te  = res[res.split == "test"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tr["year"], tr["accuracy"], "o-", color="steelblue",
            label="XGBoost (train — in-sample)")
    ax.plot(te["year"], te["accuracy"], "o-", color="darkorange",
            label="XGBoost (test — out-of-sample)")
    ax.plot(res["year"], res["baseline"], "s--", color="gray",
            label="Baseline (higher rank wins)")
    ax.axvline(cfg.train_year_end + 0.5, color="black", linestyle=":",
               label="Train / Test split")
    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model vs Baseline Accuracy by Year")
    ax.legend()
    ax.set_ylim(0.45, 0.82)
    _savefig(os.path.join(out_dir, "yearly_accuracy.png"))


def plot_all(model, X_te, y_te, probs, meta, data, cfg, use_odds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tasks = [
        ("roc_curve.png",              lambda: plot_roc(y_te, probs, out_dir)),
        ("calibration.png",            lambda: plot_calibration(y_te, probs, out_dir)),
        ("prob_hist.png",              lambda: plot_prob_histogram(y_te, probs, out_dir)),
        ("confusion_matrix.png",       lambda: plot_confusion_matrix(y_te, probs, out_dir)),
        ("precision_recall.png",       lambda: plot_precision_recall(y_te, probs, out_dir)),
        ("feature_importance.png",     lambda: plot_feature_importance(model, out_dir)),
        ("accuracy_by_category.png",   lambda: plot_accuracy_by_category(y_te, probs, meta, out_dir)),
        ("accuracy_by_confidence.png", lambda: plot_accuracy_by_confidence(y_te, probs, out_dir)),
        ("yearly_accuracy.png",        lambda: plot_yearly_accuracy(data, model, cfg, use_odds, out_dir)),
    ]
    for name, fn in tasks:
        print(f"  {name}")
        fn()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Advanced XGBoost ATP match prediction model with Elo + form features."
    )
    p.add_argument("--data-path",        default="atp_tennis.csv")
    p.add_argument("--train-year-start", type=int, default=2000)
    p.add_argument("--train-year-end",   type=int, default=2020)
    p.add_argument("--test-year-start",  type=int, default=2021)
    p.add_argument("--test-year-end",    type=int, default=2026)
    p.add_argument("--use-odds",         action="store_true",
                   help="Include bookmaker odds as features")
    p.add_argument("--tune",             action="store_true",
                   help="Run RandomizedSearchCV hyperparameter tuning")
    p.add_argument("--save-model",       default="model_advanced.joblib")
    p.add_argument("--plots-dir",        default="plots_advanced")
    a = p.parse_args()
    return Config(
        data_path=a.data_path,
        train_year_start=a.train_year_start,
        train_year_end=a.train_year_end,
        test_year_start=a.test_year_start,
        test_year_end=a.test_year_end,
        use_odds=a.use_odds,
        tune=a.tune,
        save_model=a.save_model,
        plots_dir=a.plots_dir,
    )


def main():
    cfg = parse_args()

    print(f"Loading {cfg.data_path} ...")
    df = load_data(cfg.data_path)
    print(f"  {len(df):,} matches loaded  ({df['Date'].min()} → {df['Date'].max()})")

    print("Building symmetric dataset ...")
    data = build_dataset(df, use_odds=cfg.use_odds)
    print(f"  {len(data):,} rows (2× symmetric)")

    X_tr, X_te, y_tr, y_te, meta = make_splits(data, cfg, use_odds=cfg.use_odds)
    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    tune_note = " (with hyperparameter search)" if cfg.tune else ""
    print(f"Training XGBoost{tune_note} ...")
    model = build_and_fit_model(X_tr, y_tr, use_odds=cfg.use_odds, tune=cfg.tune)

    acc, auc, brier, probs = evaluate(model, X_te, y_te)
    baseline_acc = accuracy_score(y_te, (X_te["rank_diff"] < 0).astype(int))

    print(f"\n{'─'*45}")
    print(f"Test accuracy  : {acc:.4f}   "
          f"(baseline {baseline_acc:.4f}, Δ{acc - baseline_acc:+.4f})")
    print(f"ROC AUC        : {auc:.4f}")
    print(f"Brier score    : {brier:.4f}  (lower = better, 0.25 = random)")
    print(f"{'─'*45}\n")

    if cfg.save_model:
        joblib.dump(model, cfg.save_model)
        print(f"Saved model → {cfg.save_model}")

    if cfg.plots_dir:
        print(f"Generating plots in {cfg.plots_dir}/ ...")
        plot_all(model, X_te, y_te, probs, meta, data, cfg, cfg.use_odds, cfg.plots_dir)
        print("Done.")


if __name__ == "__main__":
    main()
