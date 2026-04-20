#!/usr/bin/env python3
"""
model_6.py  -  Final Model: XGBoost + Elo + Serve Stats + Playstyle
====================================================================
Combines the strongest signals from Models 2, 3, and 5.
Model 4 (injuries) is excluded - it added noise with no predictive lift.

WHAT THIS MODEL ADDS vs MODEL 2 (baseline)
--------------------------------------------------------------------
Group H - Serve Statistics (from Model 3):
  first_serve_pct_diff         % first serves landing in (p1 - p2)
  first_serve_win_pct_diff     % points won on 1st serve (p1 - p2)
  second_serve_win_pct_diff    % points won on 2nd serve (p1 - p2)
  serve_dominance_diff         Composite weighted serve win-point %

Group I - Playstyle Matchups (from Model 5):
  p1_style / p2_style          Player style categories (categorical)
  style_matchup                Directed matchup string "A vs B"
  style_pair                   Undirected sorted pair "A vs B"
  same_style                   Binary: 1 if both players share a style

WHY THESE FEATURES TOGETHER:
  Serve data captures objective serve quality that Elo misses - critical
  on grass and fast hard courts. Playstyle captures tactical matchup
  advantages that stats alone miss (e.g. Counterpuncher vs Baseliner).
  Together they cover two independent dimensions of skill that the rank/
  Elo/form stack cannot encode.

ANTI-OVERFITTING VALIDATION:
  Beyond the standard walk-forward, this model is tested on FOUR
  independent holdout splits to verify temporal generalization:
    Split A: train 2014-2017  -> test 2018-2020
    Split B: train 2014-2019  -> test 2020-2022
    Split C: train 2014-2021  -> test 2022-2024
    Split D: train 2014-2020  -> test 2021-2024  (main, matches prior models)
  Consistent accuracy across all splits confirms the model is not overfit.

Feature groups (27 numeric + 11 categorical):
  A  Rank/skill:          rank_diff, pts_diff, rank_ratio, log_rank_diff
  B  Elo:                 elo_diff, elo_surf_diff, surface_specialist_diff
  C  Rolling form:        form_short_diff, form_long_diff, surf_form_diff, form_trend_diff
  D  H2H:                 h2h_rate, h2h_n
  E  Rest/experience:     days_rest_diff, experience_diff
  F  Player attributes:   age_diff, ht_diff
  G  Surface history:     surf_win_rate_diff, surf_exp_diff
  H  Serve stats:         first_serve_pct_diff, first_serve_win_pct_diff,
                          second_serve_win_pct_diff, serve_dominance_diff
  I  Playstyle:           same_style (numeric)
  Categorical:            surface, series, court, round, best_of,
                          p1_hand, p2_hand, p1_style, p2_style,
                          style_matchup, style_pair

Data sources:
  data/processed/atp_tennis.csv
  data/initial/atp_matches_YYYY.csv  (x11)
  data/initial/player_surface_results.csv
  data/serve_data/gemini_serve_*.csv  (x3)
  data/playstyle_reports/*.json

Usage:
  cd CSC410Pres/models/model_6
  python model_6.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
import warnings
from collections import defaultdict, deque

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(ROOT, "..", "..", "data")
INIT_DIR      = os.path.join(DATA_DIR, "initial")
PROC_DIR      = os.path.join(DATA_DIR, "processed")
SERVE_DIR     = os.path.join(DATA_DIR, "serve_data")
PLAYSTYLE_DIR = os.path.join(DATA_DIR, "playstyle_reports")
PLOTS_DIR     = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -- Elo / form constants -------------------------------------------------------
ELO_BASE   = 1500.0
ELO_K      = 32.0
ELO_K_SURF = 24.0
FORM_SHORT = 5
FORM_LONG  = 20

STYLE_ORDER   = ["Aggressive Baseliner", "All-Court Player", "Counterpuncher", "Serve-and-Volleyer"]
STYLE_UNKNOWN = "Unknown"
N_CLUSTERS    = 6   # data-driven playstyle clusters from match stats

# -- Feature column definitions -------------------------------------------------
# Groups A-G: Model 2 base
NUMERIC_M2 = [
    "rank_diff", "pts_diff", "rank_ratio", "log_rank_diff",          # A
    "elo_diff", "elo_surf_diff", "surface_specialist_diff",            # B
    "form_short_diff", "form_long_diff", "surf_form_diff",             # C
    "form_trend_diff",
    "h2h_rate", "h2h_n",                                               # D
    "days_rest_diff", "experience_diff",                               # E
    "age_diff", "ht_diff",                                             # F
    "surf_win_rate_diff", "surf_exp_diff",                             # G
]
CAT_M2 = ["surface", "series", "court", "round", "best_of", "p1_hand", "p2_hand"]

# Group H: Serve stats (Model 3)
SERVE_NUMERIC = [
    "first_serve_pct_diff",
    "first_serve_win_pct_diff",
    "second_serve_win_pct_diff",
    "serve_dominance_diff",
]

# Group I: Playstyle (Model 5) + style x surface interaction
STYLE_NUMERIC = ["same_style"]
STYLE_CAT     = ["p1_style", "p2_style", "style_matchup", "style_pair",
                 "style_surface_pair"]   # NEW: directed style-pair x surface

# Group J: Data-driven clusters from match stats (KMeans on serve+return profile)
CLUSTER_NUMERIC = ["same_cluster"]
CLUSTER_CAT     = ["p1_cluster", "p2_cluster", "cluster_matchup", "cluster_pair"]

# Full Model 6 feature set
NUMERIC_COLS = NUMERIC_M2 + SERVE_NUMERIC + STYLE_NUMERIC + CLUSTER_NUMERIC
CAT_COLS     = CAT_M2 + STYLE_CAT + CLUSTER_CAT

# Logistic baseline (mirrors model_1 / model_2 baseline)
BASELINE_NUM = ["rank_diff", "log_rank_diff", "rank_ratio", "pts_diff", "age_diff", "h2h_rate"]
BASELINE_CAT = ["surface", "series", "round", "best_of"]

# -- Holdout splits for anti-overfitting validation ----------------------------
HOLDOUT_SPLITS = [
    {"name": "Split A (train <=2017 / test 2018-20)", "train_end": 2017, "test_start": 2018, "test_end": 2020},
    {"name": "Split B (train <=2019 / test 2020-22)", "train_end": 2019, "test_start": 2020, "test_end": 2022},
    {"name": "Split C (train <=2021 / test 2022-24)", "train_end": 2021, "test_start": 2022, "test_end": 2024},
    {"name": "Split D (train <=2020 / test 2021-24)", "train_end": 2020, "test_start": 2021, "test_end": 2024},
]
MAIN_SPLIT_IDX = 3  # Split D is the main/comparable split

# -- Group colour maps ----------------------------------------------------------
GROUP_COLORS_PREFIX = [
    ("num__elo",                "darkorange"),
    ("num__surf_form",          "steelblue"),
    ("num__surface_specialist", "darkorange"),
    ("num__form",               "steelblue"),
    ("num__h2h",                "seagreen"),
    ("num__rank",               "mediumpurple"),
    ("num__pts",                "mediumpurple"),
    ("num__log",                "mediumpurple"),
    ("num__days",               "dimgray"),
    ("num__exp",                "dimgray"),
    ("num__age",                "orchid"),
    ("num__ht",                 "orchid"),
    ("num__surf_win",           "teal"),
    ("num__surf_exp",           "teal"),
    ("num__first_serve",        "crimson"),
    ("num__second_serve",       "crimson"),
    ("num__serve_dominance",    "crimson"),
    ("num__same_style",          "darkcyan"),
    ("cat__p1_style",            "darkcyan"),
    ("cat__p2_style",            "darkcyan"),
    ("cat__style_matchup",       "darkcyan"),
    ("cat__style_pair",          "darkcyan"),
    ("cat__style_surface_pair",  "darkcyan"),
    ("num__same_cluster",        "mediumvioletred"),
    ("cat__p1_cluster",          "mediumvioletred"),
    ("cat__p2_cluster",          "mediumvioletred"),
    ("cat__cluster_matchup",     "mediumvioletred"),
    ("cat__cluster_pair",        "mediumvioletred"),
]

GROUP_LABELS = {
    "A - Rank/skill":         ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"],
    "B - Elo":                ["elo_diff", "elo_surf_diff", "surface_specialist_diff"],
    "C - Rolling form":       ["form_short_diff", "form_long_diff", "surf_form_diff", "form_trend_diff"],
    "D - H2H":                ["h2h_rate", "h2h_n"],
    "E - Rest/experience":    ["days_rest_diff", "experience_diff"],
    "F - Player attributes":  ["age_diff", "ht_diff"],
    "G - Surface history":    ["surf_win_rate_diff", "surf_exp_diff"],
    "H - Serve stats":        ["first_serve_pct_diff", "first_serve_win_pct_diff",
                               "second_serve_win_pct_diff", "serve_dominance_diff"],
    "I - Playstyle":          ["same_style", "p1_style", "p2_style", "style_matchup",
                               "style_pair", "style_surface_pair"],
    "J - Data clusters":      ["same_cluster", "p1_cluster", "p2_cluster",
                               "cluster_matchup", "cluster_pair"],
}
GROUP_COLORS_BAR = {

    "A - Rank/skill":        "mediumpurple",
    "B - Elo":               "darkorange",
    "C - Rolling form":      "steelblue",
    "D - H2H":               "seagreen",
    "E - Rest/experience":   "dimgray",
    "F - Player attributes": "orchid",
    "G - Surface history":   "teal",
    "H - Serve stats":       "crimson",
    "I - Playstyle":         "darkcyan",
    "J - Data clusters":     "mediumvioletred",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def _norm(name: str) -> str:
    s = str(name).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_atp(full_name: str) -> str:
    """'First Last' or 'First Middle Last' -> 'last f.' normalized key."""
    parts = str(full_name).strip().split()
    if len(parts) >= 2:
        last_name     = " ".join(parts[1:])
        first_initial = parts[0][0]
        return _norm(f"{last_name} {first_initial}.")
    return _norm(full_name)


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n=None) -> float:
    lst = list(dq)[-n:] if n else list(dq)
    return float(np.mean(lst)) if lst else 0.5


def _feat_color(feat: str) -> str:
    for prefix, c in GROUP_COLORS_PREFIX:
        if feat.startswith(prefix):
            return c
    return "silver"


def _save(fname: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

def load_processed(proc_dir: str) -> pd.DataFrame:
    path = os.path.join(proc_dir, "atp_tennis.csv")
    df = pd.read_csv(path, low_memory=False)
    if "Best of" in df.columns:
        df = df.rename(columns={"Best of": "best_of"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["year"] = df["Date"].dt.year
    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-1, np.nan)
    for col in ["Surface", "Series", "Court", "Round"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df["best_of"] = df["best_of"].astype(str).str.strip()
    df = df.dropna(subset=["Rank_1", "Rank_2", "Surface", "Round"]).reset_index(drop=True)
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Processed: {len(df):,} matches  ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_initial_matches(init_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(init_dir, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv in {init_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["match_date"].dt.year
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "winner_ht", "loser_ht"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  Initial:   {len(df):,} matches  ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_surface_results(init_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(init_dir, "player_surface_results.csv"))
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["surface"]     = df["surface"].astype(str).str.strip().str.lower()
    for col in ["wins", "losses", "matches"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "matches" not in df.columns:
        df["matches"] = df.get("wins", 0) + df.get("losses", 0)
    df["win_rate"] = np.where(
        df["matches"] > 0,
        df.get("wins", df["matches"] * 0.5) / df["matches"], 0.5,
    )
    df["name_key"] = df["player_name"].map(_norm)
    print(f"  Surface results: {len(df):,} rows  ({df['name_key'].nunique():,} unique players)")
    return df


def load_serve_data(serve_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(serve_dir, "gemini_serve_*.csv")))
    if not files:
        raise FileNotFoundError(f"No gemini_serve_*.csv in {serve_dir}")
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    combined["player_name"] = combined["player_name"].astype(str).str.strip()
    for col in ["first_serve_pct", "first_serve_win_pct", "second_serve_win_pct"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["name_key"] = combined["player_name"].apply(_norm_atp)
    combined = (combined.groupby("name_key")[["first_serve_pct",
                                               "first_serve_win_pct",
                                               "second_serve_win_pct"]]
                         .mean().reset_index())
    print(f"  Serve stats: {len(combined):,} unique players  ({len(files)} files)")
    return combined


def load_playstyle_data(playstyle_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(playstyle_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No playstyle JSON files in {playstyle_dir}")
    canonical: dict[str, str] = {}
    aliases:   dict[str, set] = defaultdict(set)
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for row in json.load(fh):
                pname  = str(row.get("player_name", "")).strip()
                style  = str(row.get("play_style", STYLE_UNKNOWN)).strip() or STYLE_UNKNOWN
                key    = _norm(pname)
                canonical[key] = style
                aliases[key].update([key, _norm_atp(pname)])
    lookup: dict[str, str] = {}
    counts = defaultdict(int)
    for key, style in canonical.items():
        counts[style] += 1
        for alias in aliases[key]:
            if alias:
                lookup[alias] = style
    print(f"  Playstyles: {len(canonical):,} players from {len(files)} files")
    for s in STYLE_ORDER:
        print(f"    {s}: {counts.get(s, 0):,}")
    return lookup


def build_player_clusters(matches: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> tuple[dict, pd.DataFrame]:
    """
    Compute data-driven player style clusters from per-match stats.
    Uses: ace_rate, df_rate, first_in_rate, first_win_rate, second_win_rate,
          bp_save_rate, return_win_rate.
    Players are represented by career averages, then KMeans-clustered.
    Returns (lookup {name_key -> 'cluster_N'}, profile_df for plotting).
    """
    stat_cols = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                 "w_bpSaved", "w_bpFaced",
                 "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
                 "l_bpSaved", "l_bpFaced"]
    needed = ["winner_name", "loser_name"] + stat_cols
    df = matches[[c for c in needed if c in matches.columns]].copy()
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    # Winner rows: serve = w_*, return benefit = opponent lost l_svpt - l_1stWon - l_2ndWon
    for _, r in df.iterrows():
        for role, name_col, sp, fi, fw, sw, bps, bpf, osp, o1w, o2w in [
            ("w", "winner_name",
             "w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved","w_bpFaced",
             "l_svpt","l_1stWon","l_2ndWon"),
            ("l", "loser_name",
             "l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced",
             "w_svpt","w_1stWon","w_2ndWon"),
        ]:
            name = _norm(str(r.get(name_col, "")))
            if not name:
                continue
            svpt = r.get(sp, np.nan)
            if not (isinstance(svpt, float) and svpt > 0):
                continue
            first_in = r.get(fi, np.nan)
            second_pts = svpt - first_in if (isinstance(first_in, float) and first_in >= 0) else np.nan
            osp_v = r.get(osp, np.nan)
            o_ret_won = (osp_v - r.get(o1w, 0) - r.get(o2w, 0)) if isinstance(osp_v, float) and osp_v > 0 else np.nan
            bpf_v = r.get(bpf, np.nan)
            rows.append({
                "name_key":        name,
                "ace_rate":        r.get("w_ace" if role=="w" else "l_ace", np.nan) / svpt,
                "df_rate":         r.get("w_df"  if role=="w" else "l_df",  np.nan) / svpt,
                "first_in_rate":   first_in / svpt if isinstance(first_in, float) else np.nan,
                "first_win_rate":  r.get(fw, np.nan) / first_in if isinstance(first_in, float) and first_in > 0 else np.nan,
                "second_win_rate": r.get(sw, np.nan) / second_pts if isinstance(second_pts, float) and second_pts > 0 else np.nan,
                "bp_save_rate":    r.get(bps, np.nan) / bpf_v if isinstance(bpf_v, float) and bpf_v > 0 else np.nan,
                "return_win_rate": o_ret_won / osp_v if isinstance(o_ret_won, float) and isinstance(osp_v, float) and osp_v > 0 else np.nan,
            })

    stat_df = pd.DataFrame(rows)
    CLUSTER_STATS = ["ace_rate", "df_rate", "first_in_rate", "first_win_rate",
                     "second_win_rate", "bp_save_rate", "return_win_rate"]
    player_stats = (stat_df.groupby("name_key")[CLUSTER_STATS]
                           .mean()
                           .dropna(thresh=4)
                           .reset_index())

    # Scale and cluster
    X = player_stats[CLUSTER_STATS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.mean()))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    player_stats["cluster_id"] = km.fit_predict(X_scaled)
    player_stats["cluster"] = player_stats["cluster_id"].apply(lambda x: f"cluster_{x}")

    # Build lookup
    lookup = dict(zip(player_stats["name_key"], player_stats["cluster"]))

    # Build human-readable profile df for plots (cluster centroids in original scale)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                           columns=CLUSTER_STATS)
    centers["cluster"] = [f"cluster_{i}" for i in range(n_clusters)]

    # Label each cluster by its dominant characteristic
    def _label(row):
        if row["ace_rate"] > centers["ace_rate"].median() * 1.3:
            return f"cluster_{int(row.name)} (Big Serve)"
        if row["return_win_rate"] > centers["return_win_rate"].median() * 1.1:
            return f"cluster_{int(row.name)} (Returner)"
        if row["first_win_rate"] > centers["first_win_rate"].median() * 1.05:
            return f"cluster_{int(row.name)} (1st Serve Dom.)"
        if row["second_win_rate"] < centers["second_win_rate"].median() * 0.92:
            return f"cluster_{int(row.name)} (2nd Serve Vuln.)"
        return f"cluster_{int(row.name)} (All-Round)"

    centers["label"] = [_label(centers.iloc[i]) for i in range(len(centers))]
    sizes = player_stats["cluster"].value_counts().to_dict()
    centers["size"] = centers["cluster"].map(sizes)

    print(f"  Data-driven clusters: {n_clusters} clusters from {len(player_stats):,} players")
    for _, row in centers.iterrows():
        print(f"    {row['cluster']} ({row['size']:,} players): "
              f"ace={row['ace_rate']:.3f}  ret={row['return_win_rate']:.3f}  "
              f"1stW={row['first_win_rate']:.3f}")
    return lookup, centers


# ==============================================================================
# 2.  STATIC LOOKUPS
# ==============================================================================

def build_player_attrs(matches: pd.DataFrame) -> dict:
    parts = []
    for role, name_col, hand_col, ht_col, age_col in [
        ("w", "winner_name", "winner_hand", "winner_ht", "winner_age"),
        ("l", "loser_name",  "loser_hand",  "loser_ht",  "loser_age"),
    ]:
        parts.append(pd.DataFrame({
            "name_key":   matches[name_col].map(_norm),
            "hand":       matches.get(hand_col, pd.Series("U", index=matches.index)),
            "ht":         matches.get(ht_col,   pd.Series(np.nan, index=matches.index)),
            "birth_year": matches["year"] - matches[age_col],
        }))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["name_key"] != ""]
    def _mode(s):
        m = s.dropna().mode()
        return str(m.iloc[0]) if len(m) else "U"
    result: dict[str, dict] = {}
    for key, grp in df.groupby("name_key"):
        result[key] = {
            "hand":       _mode(grp["hand"]),
            "ht":         float(grp["ht"].median())         if grp["ht"].notna().any()         else np.nan,
            "birth_year": float(grp["birth_year"].median()) if grp["birth_year"].notna().any() else np.nan,
        }
    print(f"  Player attrs: {len(result):,} players indexed")
    return result


def build_surface_lookup(surf_df: pd.DataFrame) -> dict:
    lookup: dict[tuple, dict] = {}
    for _, row in surf_df.iterrows():
        lookup[(row["name_key"], row["surface"])] = {
            "win_rate":  float(row["win_rate"]),
            "n_matches": int(row["matches"]),
        }
    print(f"  Surface lookup: {len(lookup):,} (player, surface) entries")
    return lookup


def build_serve_lookup(serve_df: pd.DataFrame) -> dict:
    lookup: dict[str, dict] = {}
    for _, row in serve_df.iterrows():
        lookup[row["name_key"]] = {
            "first_serve_pct":      float(row["first_serve_pct"]),
            "first_serve_win_pct":  float(row["first_serve_win_pct"]),
            "second_serve_win_pct": float(row["second_serve_win_pct"]),
        }
    print(f"  Serve lookup: {len(lookup):,} players")
    return lookup


# ==============================================================================
# 3.  DYNAMIC FEATURES
# ==============================================================================

def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    elo_all  = defaultdict(lambda: ELO_BASE)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all  = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_dt: dict = {}
    exp_cnt  = defaultdict(int)
    h2h_wins = defaultdict(int)

    feat_rows = []
    for row in df.itertuples(index=False):
        p1 = getattr(row, "Player_1", None)
        p2 = getattr(row, "Player_2", None)
        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win    = getattr(row, "Winner", None)
        surf   = str(getattr(row, "Surface", "unknown") or "unknown").lower()
        date   = getattr(row, "Date", None)
        p1_won = (win == p1)

        p1_elo  = elo_all[p1];          p2_elo  = elo_all[p2]
        p1_eloS = elo_surf[p1][surf];   p2_eloS = elo_surf[p2][surf]

        p1_fS = _form(res_all[p1], FORM_SHORT);  p2_fS = _form(res_all[p2], FORM_SHORT)
        p1_fL = _form(res_all[p1]);               p2_fL = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf]);         p2_sf = _form(res_surf[p2][surf])

        h2h_p1    = h2h_wins[(p1, p2)]
        h2h_p2    = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        h2h_rate  = (h2h_p1 + 1.0) / (h2h_total + 2.0)

        p1_days = float((date - last_dt[p1]).days) if p1 in last_dt else 30.0
        p2_days = float((date - last_dt[p2]).days) if p2 in last_dt else 30.0

        feat_rows.append({
            "p1_elo_pre":      p1_elo,  "p2_elo_pre":      p2_elo,
            "p1_eloS_pre":     p1_eloS, "p2_eloS_pre":     p2_eloS,
            "p1_fS_pre":       p1_fS,   "p2_fS_pre":       p2_fS,
            "p1_fL_pre":       p1_fL,   "p2_fL_pre":       p2_fL,
            "p1_sf_pre":       p1_sf,   "p2_sf_pre":       p2_sf,
            "p1_h2h_rate_pre": h2h_rate,
            "h2h_n_pre":       float(h2h_total),
            "p1_days_pre":     p1_days, "p2_days_pre":     p2_days,
            "p1_exp_pre":      float(exp_cnt[p1]),
            "p2_exp_pre":      float(exp_cnt[p2]),
        })

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
            last_dt[p1] = date
            last_dt[p2] = date
        exp_cnt[p1] += 1
        exp_cnt[p2] += 1

    return pd.concat([df, pd.DataFrame(feat_rows, index=df.index)], axis=1)


# ==============================================================================
# 4.  DATASET CONSTRUCTION
# ==============================================================================

def _make_rows(df, flip, player_attrs, surf_lookup, serve_lookup, playstyle_lookup, cluster_lookup):
    def _v(a, b):
        col = b if flip else a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    p1_names = df["Player_2" if flip else "Player_1"].values
    p2_names = df["Player_1" if flip else "Player_2"].values
    label    = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values
    h2h_rate = (
        1.0 - df["p1_h2h_rate_pre"].values if flip
        else df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    surf_vals   = df["Surface"].values
    match_years = df["year"].values.astype(float)

    def _flt(names, key):
        return np.array([float(player_attrs.get(_norm(n), {}).get(key, np.nan))
                         for n in names], dtype=float)

    def _srf(names, key, default):
        return np.array([
            float(surf_lookup.get((_norm(n), str(s or "").lower()), {}).get(key, default))
            for n, s in zip(names, surf_vals)
        ], dtype=float)

    def _srv(names, key):
        return np.array([serve_lookup.get(_norm(n), {}).get(key, np.nan)
                         for n in names], dtype=float)

    def _style(names):
        return np.array([playstyle_lookup.get(_norm(n), STYLE_UNKNOWN) for n in names],
                        dtype=object)

    def _cluster(names):
        return np.array([cluster_lookup.get(_norm(n), "cluster_unknown") for n in names],
                        dtype=object)

    p1_by   = _flt(p1_names, "birth_year");  p2_by   = _flt(p2_names, "birth_year")
    p1_ht   = _flt(p1_names, "ht");          p2_ht   = _flt(p2_names, "ht")
    p1_hand = np.array([player_attrs.get(_norm(n), {}).get("hand", "U") for n in p1_names])
    p2_hand = np.array([player_attrs.get(_norm(n), {}).get("hand", "U") for n in p2_names])

    p1_sty  = _style(p1_names)
    p2_sty  = _style(p2_names)
    matchup = np.array([f"{a} vs {b}" for a, b in zip(p1_sty, p2_sty)], dtype=object)
    pair    = np.array([" vs ".join(sorted([a, b])) for a, b in zip(p1_sty, p2_sty)], dtype=object)

    p1_clu  = _cluster(p1_names)
    p2_clu  = _cluster(p2_names)
    clu_matchup = np.array([f"{a} vs {b}" for a, b in zip(p1_clu, p2_clu)], dtype=object)
    clu_pair    = np.array([" vs ".join(sorted([a, b])) for a, b in zip(p1_clu, p2_clu)], dtype=object)

    return pd.DataFrame({
        "year":    df["year"].values,
        "p1_rank": _v("Rank_1", "Rank_2"), "p2_rank": _v("Rank_2", "Rank_1"),
        "p1_pts":  _v("Pts_1",  "Pts_2"),  "p2_pts":  _v("Pts_2",  "Pts_1"),
        "p1_elo":  _v("p1_elo_pre",  "p2_elo_pre"),
        "p2_elo":  _v("p2_elo_pre",  "p1_elo_pre"),
        "p1_eloS": _v("p1_eloS_pre", "p2_eloS_pre"),
        "p2_eloS": _v("p2_eloS_pre", "p1_eloS_pre"),
        "p1_fS":   _v("p1_fS_pre",   "p2_fS_pre"),
        "p2_fS":   _v("p2_fS_pre",   "p1_fS_pre"),
        "p1_fL":   _v("p1_fL_pre",   "p2_fL_pre"),
        "p2_fL":   _v("p2_fL_pre",   "p1_fL_pre"),
        "p1_sf":   _v("p1_sf_pre",   "p2_sf_pre"),
        "p2_sf":   _v("p2_sf_pre",   "p1_sf_pre"),
        "h2h_rate": h2h_rate,
        "h2h_n":    df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
        "p1_days": _v("p1_days_pre", "p2_days_pre"),
        "p2_days": _v("p2_days_pre", "p1_days_pre"),
        "p1_exp":  _v("p1_exp_pre",  "p2_exp_pre"),
        "p2_exp":  _v("p2_exp_pre",  "p1_exp_pre"),
        "p1_age":  match_years - p1_by,  "p2_age":  match_years - p2_by,
        "p1_ht":   p1_ht,                "p2_ht":   p2_ht,
        "p1_hand": p1_hand,              "p2_hand": p2_hand,
        "p1_swr":  _srf(p1_names, "win_rate",  np.nan),
        "p2_swr":  _srf(p2_names, "win_rate",  np.nan),
        "p1_sn":   _srf(p1_names, "n_matches", 0),
        "p2_sn":   _srf(p2_names, "n_matches", 0),
        # Serve stats (Group H)
        "p1_fsp":  _srv(p1_names, "first_serve_pct"),
        "p2_fsp":  _srv(p2_names, "first_serve_pct"),
        "p1_fswp": _srv(p1_names, "first_serve_win_pct"),
        "p2_fswp": _srv(p2_names, "first_serve_win_pct"),
        "p1_sswp": _srv(p1_names, "second_serve_win_pct"),
        "p2_sswp": _srv(p2_names, "second_serve_win_pct"),
        # Playstyle (Group I) + style x surface interaction
        "p1_style":           p1_sty,
        "p2_style":           p2_sty,
        "style_matchup":      matchup,
        "style_pair":         pair,
        "style_surface_pair": np.array([f"{p}_{s}" for p, s in
                                         zip(pair, df["Surface"].values)], dtype=object),
        "p1_has_style":  (p1_sty != STYLE_UNKNOWN).astype(int),
        "p2_has_style":  (p2_sty != STYLE_UNKNOWN).astype(int),
        "p1_has_serve":  (~np.isnan(_srv(p1_names, "first_serve_pct"))).astype(int),
        "p2_has_serve":  (~np.isnan(_srv(p2_names, "first_serve_pct"))).astype(int),
        # Data-driven clusters (Group J)
        "p1_cluster":      p1_clu,
        "p2_cluster":      p2_clu,
        "cluster_matchup": clu_matchup,
        "cluster_pair":    clu_pair,
        "surface": df["Surface"].values,
        "series":  df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
        "court":   df["Court"].values  if "Court"  in df.columns else np.full(len(df), "unknown"),
        "round":   df["Round"].values,
        "best_of": df["best_of"].values,
        "label":   label,
    })


def build_dataset(df, player_attrs, surf_lookup, serve_lookup, playstyle_lookup, cluster_lookup):
    df = df.copy()
    df["p1_wins"] = (df["Player_1"] == df["Winner"]).astype(int)
    data = pd.concat(
        [_make_rows(df, False, player_attrs, surf_lookup, serve_lookup, playstyle_lookup, cluster_lookup),
         _make_rows(df, True,  player_attrs, surf_lookup, serve_lookup, playstyle_lookup, cluster_lookup)],
        ignore_index=True,
    )

    # Group A
    data["rank_diff"]               = data["p1_rank"]  - data["p2_rank"]
    data["pts_diff"]                = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"]              = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"]           = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])
    # Group B
    data["elo_diff"]                = data["p1_elo"]  - data["p2_elo"]
    data["elo_surf_diff"]           = data["p1_eloS"] - data["p2_eloS"]
    data["surface_specialist_diff"] = data["elo_surf_diff"] - data["elo_diff"]
    # Group C
    data["form_short_diff"]         = data["p1_fS"]   - data["p2_fS"]
    data["form_long_diff"]          = data["p1_fL"]   - data["p2_fL"]
    data["surf_form_diff"]          = data["p1_sf"]   - data["p2_sf"]
    data["form_trend_diff"]         = data["form_short_diff"] - data["form_long_diff"]
    # Group E
    data["days_rest_diff"]          = data["p1_days"] - data["p2_days"]
    data["experience_diff"]         = data["p1_exp"]  - data["p2_exp"]
    # Group F
    data["age_diff"]                = data["p1_age"]  - data["p2_age"]
    data["ht_diff"]                 = data["p1_ht"]   - data["p2_ht"]
    # Group G
    data["surf_win_rate_diff"]      = data["p1_swr"]  - data["p2_swr"]
    data["surf_exp_diff"]           = data["p1_sn"]   - data["p2_sn"]
    # Group H - Serve stats
    data["first_serve_pct_diff"]      = data["p1_fsp"]  - data["p2_fsp"]
    data["first_serve_win_pct_diff"]  = data["p1_fswp"] - data["p2_fswp"]
    data["second_serve_win_pct_diff"] = data["p1_sswp"] - data["p2_sswp"]
    p1_dom = ((data["p1_fsp"] / 100.0) * (data["p1_fswp"] / 100.0) +
              (1.0 - data["p1_fsp"] / 100.0) * (data["p1_sswp"] / 100.0))
    p2_dom = ((data["p2_fsp"] / 100.0) * (data["p2_fswp"] / 100.0) +
              (1.0 - data["p2_fsp"] / 100.0) * (data["p2_sswp"] / 100.0))
    data["serve_dominance_diff"] = p1_dom - p2_dom
    # Group I - Playstyle
    data["same_style"]           = (data["p1_style"] == data["p2_style"]).astype(int)
    data["playstyle_coverage"]   = data["p1_has_style"] + data["p2_has_style"]
    data["serve_coverage"]       = data["p1_has_serve"] + data["p2_has_serve"]
    # Group J - Data-driven clusters
    data["same_cluster"]         = (data["p1_cluster"] == data["p2_cluster"]).astype(int)

    print(f"  Dataset: {len(data):,} rows  ({int(data['year'].min())}-{int(data['year'].max())})")
    return data


# ==============================================================================
# 5.  MODEL PIPELINES
# ==============================================================================

def _preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                               ("cat", cat_pipe, cat_cols)])


def make_m6_pipeline() -> CalibratedClassifierCV:
    """Model 6: XGBoost with all features (Elo + Serve + Playstyle), isotonic calibration."""
    pre = _preprocessor(NUMERIC_COLS, CAT_COLS)
    xgb = XGBClassifier(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.70,
        min_child_weight=20,
        gamma=0.1,
        reg_alpha=0.10,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]),
                                  cv=3, method="isotonic")


def make_m2_pipeline() -> CalibratedClassifierCV:
    """Model 2 equivalent for baseline comparison."""
    pre = _preprocessor(NUMERIC_M2, CAT_M2)
    xgb = XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=15,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]),
                                  cv=3, method="isotonic")


def make_baseline_pipeline() -> Pipeline:
    pre = _preprocessor(BASELINE_NUM, BASELINE_CAT)
    lr  = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return Pipeline([("pre", pre), ("clf", lr)])


def fit_eval(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return accuracy_score(y_te, preds), roc_auc_score(y_te, probs), \
           brier_score_loss(y_te, probs), probs


# ==============================================================================
# 6.  VALIDATION
# ==============================================================================

def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_yr in sorted(data["year"].unique()):
        if test_yr < 2016 or test_yr > 2024:
            continue
        tr = data[data["year"] < test_yr]
        te = data[data["year"] == test_yr]
        if len(tr) < 100 or len(te) < 50:
            continue

        y_tr = tr["label"];  y_te = te["label"]

        m6 = make_m6_pipeline()
        acc6, auc6, _, _ = fit_eval(m6, tr[NUMERIC_COLS + CAT_COLS], y_tr,
                                        te[NUMERIC_COLS + CAT_COLS], y_te)
        mb = make_baseline_pipeline()
        acc_b, auc_b, _, _ = fit_eval(mb, tr[BASELINE_NUM + BASELINE_CAT], y_tr,
                                          te[BASELINE_NUM + BASELINE_CAT], y_te)
        base = accuracy_score(y_te, (te["rank_diff"] < 0).astype(int))

        rows.append(dict(test_year=test_yr, acc_m6=acc6, auc_m6=auc6,
                         acc_lr=acc_b, auc_lr=auc_b, rank_baseline=base, n=len(te)))
        print(f"    {test_yr}: M6={acc6:.4f}  LR={acc_b:.4f}  "
              f"rank={base:.4f}  (n={len(te):,})")
    return pd.DataFrame(rows)


def multi_holdout_validation(data: pd.DataFrame) -> list[dict]:
    """Train/test on four different year ranges to verify temporal robustness."""
    results = []
    for split in HOLDOUT_SPLITS:
        tr = data[(data["year"] >= 2014) & (data["year"] <= split["train_end"])]
        te = data[(data["year"] >= split["test_start"]) & (data["year"] <= split["test_end"])]
        if len(tr) < 200 or len(te) < 50:
            results.append({**split, "acc_m6": np.nan, "auc_m6": np.nan,
                             "acc_m2": np.nan, "n_train": len(tr), "n_test": len(te)})
            continue

        y_tr = tr["label"];  y_te = te["label"]

        m6 = make_m6_pipeline()
        acc6, auc6, brier6, _ = fit_eval(m6, tr[NUMERIC_COLS + CAT_COLS], y_tr,
                                              te[NUMERIC_COLS + CAT_COLS], y_te)
        m2 = make_m2_pipeline()
        acc2, auc2, brier2, _ = fit_eval(m2, tr[NUMERIC_M2 + CAT_M2], y_tr,
                                              te[NUMERIC_M2 + CAT_M2], y_te)
        rank_b = accuracy_score(y_te, (te["rank_diff"] < 0).astype(int))

        print(f"  {split['name']}: M6={acc6:.4f}  M2={acc2:.4f}  "
              f"rank={rank_b:.4f}  "
              f"(n_train={len(tr):,}, n_test={len(te):,})")
        results.append({**split, "acc_m6": acc6, "auc_m6": auc6, "brier_m6": brier6,
                         "acc_m2": acc2, "auc_m2": auc2, "brier_m2": brier2,
                         "rank_baseline": rank_b, "n_train": len(tr), "n_test": len(te)})
    return results


# ==============================================================================
# 7.  FEATURE IMPORTANCE EXTRACTION
# ==============================================================================

def _extract_importances(model):
    try:
        pipes = ([cc.estimator for cc in model.calibrated_classifiers_]
                 if hasattr(model, "calibrated_classifiers_") else [model])
        imps  = [p.named_steps["clf"].feature_importances_
                 for p in pipes if hasattr(p, "named_steps")]
        if not imps:
            return None
        avg   = np.mean(imps, axis=0)
        names = list(pipes[0].named_steps["pre"].get_feature_names_out())
        return names, avg
    except Exception as exc:
        print(f"  Warning: importance extraction failed: {exc}")
        return None


# ==============================================================================
# 8.  PLOTS
# ==============================================================================

def plot_roc_curve(y_te, probs, acc, auc):
    fpr, tpr, _ = roc_curve(y_te, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="darkgreen",
            label=f"Model 6 Final (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Model 6 (Final)\n(Test 2021-2024  |  Accuracy={acc:.4f})")
    ax.legend()
    _save("roc_curve.png")


def plot_calibration(y_te, probs):
    fop, mpv = calibration_curve(y_te, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="darkgreen", lw=2, label="Model 6")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve - Model 6 (Final)")
    ax.legend()
    _save("calibration.png")


def plot_confusion_matrix(y_te, probs):
    cm = confusion_matrix(y_te, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Loss", "Predicted Win"], fontsize=11)
    ax.set_yticklabels(["Actual Loss", "Actual Win"], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=16,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title("Confusion Matrix - Model 6 Final (Test 2021-2024)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    _save("confusion_matrix.png")


def plot_feature_importance(model):
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(os.path.join(PLOTS_DIR, "feature_importance.csv"), index=False)
    top = df.head(28).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(top["feature"], top["importance"],
            color=[_feat_color(f) for f in top["feature"]])
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Top 28 Feature Importances - Model 6 Final\n"
                 "(crimson = serve stats, darkcyan = playstyle)")
    ax.legend(handles=[
        mpatches.Patch(facecolor="darkorange",   label="Elo (B)"),
        mpatches.Patch(facecolor="steelblue",    label="Rolling form (C)"),
        mpatches.Patch(facecolor="seagreen",     label="H2H (D)"),
        mpatches.Patch(facecolor="mediumpurple", label="Rank/points (A)"),
        mpatches.Patch(facecolor="teal",         label="Surface history (G)"),
        mpatches.Patch(facecolor="crimson",      label="Serve stats (H) - NEW"),
        mpatches.Patch(facecolor="darkcyan",        label="Playstyle (I) - NEW"),
        mpatches.Patch(facecolor="mediumvioletred", label="Data clusters (J) - NEW"),
        mpatches.Patch(facecolor="dimgray",         label="Rest/experience (E)"),
        mpatches.Patch(facecolor="orchid",          label="Player attributes (F)"),
    ], loc="lower right", fontsize=8)
    _save("feature_importance.png")


def plot_walk_forward(wf: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.plot(wf.test_year, wf.acc_m6,        "o-",  color="darkgreen",  lw=2,
             label="Model 6 - Final (Elo + Serve + Playstyle)")
    ax1.plot(wf.test_year, wf.rank_baseline, "^:",  color="gray",        lw=1.5,
             label="Rank baseline")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Walk-Forward Validation - Accuracy (Model 6 Final)")
    ax1.legend(); ax1.set_ylim(0.55, 0.82); ax1.grid(axis="y", alpha=0.3)
    ax2.plot(wf.test_year, wf.auc_m6, "o-", color="darkgreen", lw=2)
    ax2.set_ylabel("ROC AUC"); ax2.set_xlabel("Test Year")
    ax2.set_title("Walk-Forward Validation - ROC AUC (Model 6 Final)")
    ax2.set_ylim(0.60, 0.88); ax2.grid(axis="y", alpha=0.3)
    for _, r in wf.iterrows():
        ax1.annotate(f"n={int(r.n):,}", xy=(r.test_year, r.acc_m6 - 0.015),
                     ha="center", fontsize=7, color="gray")
    _save("walk_forward.png")


def plot_accuracy_by_surface(test, model):
    valid, accs, bases, ns = [], [], [], []
    for surf in sorted(s for s in test["surface"].unique()
                       if s not in ("unknown", "nan") and pd.notna(s)):
        sub = test[test["surface"] == surf]
        if len(sub) < 100:
            continue
        p = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        valid.append(surf)
        accs.append(accuracy_score(sub["label"], (p >= 0.5).astype(int)))
        bases.append(accuracy_score(sub["label"], (sub["rank_diff"] < 0).astype(int)))
        ns.append(len(sub))
    x = np.arange(len(valid)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs,  w, color="darkgreen", alpha=0.85, label="Model 6 Final")
    ax.bar(x + w/2, bases, w, color="lightgray", alpha=0.85, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.title()}\n(n={n:,})" for s, n in zip(valid, ns)])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.55, 0.82)
    ax.set_title("Accuracy by Surface - Model 6 Final vs Rank Baseline (Test 2021-2024)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1); ax.legend()
    _save("accuracy_by_surface.png")


def plot_accuracy_by_confidence(y_te, probs):
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    df["conf"]    = np.abs(df["prob"] - 0.5)
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    bins   = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True)["correct"].agg(acc="mean", n="count").reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(grp["bin"], grp["acc"], alpha=0.7, color="darkgreen", label="Accuracy")
    ax2.plot(grp["bin"], grp["n"], "o-", color="steelblue", label="Count")
    ax1.axhline(0.5, color="gray", linestyle="--")
    ax1.set_xlabel("Confidence Bucket (|P - 0.5|)")
    ax1.set_ylabel("Accuracy"); ax2.set_ylabel("Sample Count")
    ax1.set_title("Accuracy vs Prediction Confidence - Model 6 Final")
    plt.xticks(rotation=30, ha="right")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower right")
    _save("accuracy_by_confidence.png")


def plot_prob_histogram(y_te, probs):
    df = pd.DataFrame({"prob": probs, "label": y_te.values})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df.label==1]["prob"], bins=30, alpha=0.6, color="darkgreen", label="Wins (label=1)")
    ax.hist(df[df.label==0]["prob"], bins=30, alpha=0.6, color="steelblue", label="Losses (label=0)")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Predicted Win Probability"); ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution - Model 6 Final")
    ax.legend()
    _save("prob_histogram.png")


def plot_vs_baseline_roc(y_te, probs_m6, probs_lr):
    fpr6, tpr6, _ = roc_curve(y_te, probs_m6)
    fpr_b, tpr_b, _ = roc_curve(y_te, probs_lr)
    auc6  = roc_auc_score(y_te, probs_m6)
    auc_b = roc_auc_score(y_te, probs_lr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr6,  tpr6,  lw=2, color="darkgreen",
            label=f"Model 6 Final (AUC={auc6:.4f})")
    ax.plot(fpr_b, tpr_b, lw=2, color="steelblue", linestyle="--",
            label=f"Logistic baseline (AUC={auc_b:.4f})")
    ax.plot([0,1], [0,1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison - Model 6 vs Logistic Baseline (Test 2021-2024)")
    ax.fill_between(fpr6, np.interp(fpr6, fpr_b, tpr_b), tpr6,
                    alpha=0.08, color="darkgreen", label="Improvement region")
    ax.legend()
    _save("vs_baseline_roc.png")


def plot_yearly_comparison(wf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m6,        "o-",  color="darkgreen",  lw=2,
            label="Model 6 Final - Elo + Serve + Playstyle")
    ax.plot(wf.test_year, wf.acc_lr,        "s--", color="steelblue",  lw=2,
            label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:",  color="gray",       lw=1.5,
            label="Rank baseline")
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m6, alpha=0.1, color="darkgreen",
                    label="Model 6 gain over logistic")
    ax.set_xlabel("Test Year"); ax.set_ylabel("Accuracy")
    ax.set_title("Yearly Accuracy - Model 6 Final vs Baselines")
    ax.legend(); ax.set_ylim(0.55, 0.80); ax.grid(axis="y", alpha=0.3)
    for _, r in wf.iterrows():
        delta = r.acc_m6 - r.acc_lr
        ax.annotate(f"+{delta:.3f}", xy=(r.test_year, (r.acc_m6+r.acc_lr)/2),
                    ha="center", fontsize=7.5, color="darkgreen", fontweight="bold")
    _save("yearly_comparison.png")


def plot_feature_groups(model):
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    group_totals = {}
    for grp, feats in GROUP_LABELS.items():
        total = sum(imp_df[imp_df["feature"].str.contains(f, regex=False)]["importance"].sum()
                    for f in feats)
        group_totals[grp] = total
    total_imp = sum(group_totals.values()) or 1.0
    groups = list(group_totals.keys())
    pcts   = [group_totals[g] / total_imp * 100 for g in groups]
    colors = [GROUP_COLORS_BAR[g] for g in groups]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(groups, pcts, color=colors, edgecolor="white")
    ax.set_xlabel("% of Total Model Gain")
    ax.set_title("Feature Importance by Group - Model 6 Final\n"
                 "(crimson = serve stats H, darkcyan = playstyle I)")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(pcts) * 1.2)
    _save("feature_groups.png")


def plot_elo_validation(test: pd.DataFrame):
    df = test[["elo_diff", "label"]].dropna().copy()
    bin_edges  = [-600, -300, -150, -75, -25, 25, 75, 150, 300, 600]
    bin_labels = [f"{(a+b)//2:+d}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    df["bin"]  = pd.cut(df["elo_diff"], bins=bin_edges, labels=bin_labels)
    grp = (df.groupby("bin", observed=True)["label"]
             .agg(win_rate="mean", n="count").reset_index().dropna())
    midpoints  = [(a+b)/2 for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    elo_theory = [_elo_expected(1500+m, 1500) for m in midpoints]
    grp = grp.merge(pd.DataFrame({"bin": bin_labels, "elo_prob": elo_theory}),
                    on="bin", how="left")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grp["bin"], grp["win_rate"], color="darkgreen", alpha=0.75,
           label="Actual win rate")
    ax.plot(grp["bin"], grp["elo_prob"], "o--", color="steelblue", lw=2,
            label="Theoretical Elo win probability")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Elo Difference (p1 - p2)"); ax.set_ylabel("Win Rate")
    ax.set_title("Elo Signal Validation - Model 6 Final")
    for _, r in grp.iterrows():
        ax.annotate(f"n={int(r.n):,}", xy=(r.bin, r.win_rate + 0.01),
                    ha="center", fontsize=7.5, color="gray")
    ax.legend(); ax.set_ylim(0.25, 0.80)
    _save("elo_validation.png")


def plot_accuracy_by_round(test: pd.DataFrame, model):
    order  = ["r128", "r64", "r32", "r16", "qf", "sf", "f", "rr", "br"]
    rounds = [r for r in order if r in test["round"].unique()]
    rounds += [r for r in test["round"].unique()
               if r not in order and r not in ("unknown", "nan") and pd.notna(r)]
    accs, ns, valid = [], [], []
    for rnd in rounds:
        sub = test[test["round"] == rnd]
        if len(sub) < 50:
            continue
        p = model.predict_proba(sub[NUMERIC_COLS + CAT_COLS])[:, 1]
        accs.append(accuracy_score(sub["label"], (p >= 0.5).astype(int)))
        ns.append(len(sub)); valid.append(rnd.upper())
    x = np.arange(len(valid))
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, accs, color="darkgreen", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={n:,})" for r, n in zip(valid, ns)], fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.55, 0.88)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_title("Accuracy by Round - Model 6 Final (Test 2021-2024)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", fontsize=9)
    _save("accuracy_by_round.png")


def plot_rank_gap_accuracy(test: pd.DataFrame, probs):
    df = test[["rank_diff", "label"]].copy()
    df["prob"]    = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    df["abs_gap"] = df["rank_diff"].abs()
    bins   = [0, 10, 25, 50, 100, 200, 500, 2000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
    df["gap_bin"] = pd.cut(df["abs_gap"], bins=bins, labels=labels)
    grp = (df.groupby("gap_bin", observed=True)
             .agg(m6=("correct", "mean"),
                  base=("abs_gap", lambda x: ((df.loc[x.index, "rank_diff"] < 0) ==
                                               df.loc[x.index, "label"]).mean()),
                  n=("correct", "count")).reset_index())
    x = np.arange(len(grp)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, grp["m6"],   w, color="darkgreen", alpha=0.85, label="Model 6 Final")
    ax.bar(x + w/2, grp["base"], w, color="lightgray", alpha=0.85, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{str(r['gap_bin'])}\n(n={int(r.n):,})"
                        for _, r in grp.iterrows()], fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("|Rank gap| between players")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.40, 0.95)
    ax.set_title("Accuracy by Rank Gap - Model 6 Final vs Rank Baseline")
    ax.legend()
    _save("rank_gap_accuracy.png")


# -- NEW Charts -----------------------------------------------------------------

def plot_multi_holdout(holdout_results: list[dict]):
    """
    Bar chart showing Model 6 vs Model 2 accuracy across four different
    train/test year splits. Consistent accuracy = model is not overfit.
    """
    valid = [r for r in holdout_results if not np.isnan(r.get("acc_m6", np.nan))]
    if not valid:
        return
    names = [r["name"] for r in valid]
    acc6  = [r["acc_m6"] for r in valid]
    acc2  = [r["acc_m2"] for r in valid]
    rank  = [r["rank_baseline"] for r in valid]

    x = np.arange(len(names)); w = 0.28
    fig, ax = plt.subplots(figsize=(14, 6))
    b6 = ax.bar(x - w,     acc6, w, color="darkgreen", alpha=0.88, label="Model 6 Final")
    b2 = ax.bar(x,         acc2, w, color="darkorange", alpha=0.88, label="Model 2 (baseline)")
    br = ax.bar(x + w,     rank, w, color="lightgray",  alpha=0.88, label="Rank baseline")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.60, 0.80)
    ax.set_title("Anti-Overfitting Check - Model 6 Accuracy Across 4 Holdout Splits\n"
                 "(consistent accuracy across different train/test years = robust model)")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    for bars in [b6, b2, br]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate M6 gains over M2
    for i, (a6, a2) in enumerate(zip(acc6, acc2)):
        delta = a6 - a2
        sign  = "+" if delta >= 0 else ""
        color = "darkgreen" if delta >= 0 else "red"
        ax.annotate(f"{sign}{delta:.3f}", xy=(i - w/2, (a6+a2)/2),
                    ha="center", fontsize=8, color=color, fontweight="bold")
    _save("multi_holdout_comparison.png")


def plot_m6_vs_all_comparison(acc6, auc6, brier6, acc2, auc2, brier2, acc_lr, auc_lr, brier_lr):
    """Side-by-side bar chart: Model 6 vs Model 2 vs Logistic baseline."""
    models  = ["Logistic\nBaseline", "Model 2\n(Elo+Form)", "Model 6\nFinal"]
    accs    = [acc_lr,  acc2,   acc6]
    aucs    = [auc_lr,  auc2,   auc6]
    briers  = [brier_lr, brier2, brier6]
    colors  = ["steelblue", "darkorange", "darkgreen"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, vals, title, ylabel, invert in [
        (axes[0], accs,   "Accuracy (^ better)",  "Accuracy",    False),
        (axes[1], aucs,   "ROC AUC (^ better)",   "ROC AUC",     False),
        (axes[2], briers, "Brier Score (v better)", "Brier Score", True),
    ]:
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        vmin = min(vals) * 0.985; vmax = max(vals) * 1.015
        ax.set_ylim(vmin, vmax)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (vmax - vmin) * 0.01,
                    f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        delta = vals[2] - vals[1]
        sign  = "+" if delta > 0 else ""
        good  = (delta > 0 and not invert) or (delta < 0 and invert)
        ax.annotate(
            f"M6 vs M2: {sign}{delta:.4f}",
            xy=(2, vals[2]), xytext=(1.5, vmax * 0.997),
            fontsize=9, color="green" if good else "red",
            fontweight="bold", ha="center",
        )
    fig.suptitle("Model 6 Final vs Model 2 vs Logistic Baseline\n"
                 "(Test 2021-2024 - train 2014-2020)", fontsize=13, fontweight="bold")
    _save("m6_vs_models_comparison.png")


def plot_new_features_importance(model):
    """Highlight serve and playstyle features vs inherited Model 2 features."""
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(30)
    df = df.sort_values("importance", ascending=True)

    serve_kw    = ["first_serve", "second_serve", "serve_dominance"]
    style_kw    = ["style", "p1_style", "p2_style"]
    def _classify(feat):
        if any(k in feat for k in serve_kw):
            return "crimson"
        if any(k in feat for k in style_kw):
            return "darkcyan"
        return "#aaaaaa"

    colors = [_classify(f) for f in df["feature"]]
    fig, ax = plt.subplots(figsize=(12, 11))
    bars = ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Model 6 Final - New Features vs Model 2 Inherited\n"
                 "(crimson = serve stats H; darkcyan = playstyle I; grey = M2 base)")
    ax.legend(handles=[
        mpatches.Patch(facecolor="crimson",         label="Serve stats (Group H) - NEW"),
        mpatches.Patch(facecolor="darkcyan",        label="Playstyle (Group I) - NEW"),
        mpatches.Patch(facecolor="mediumvioletred", label="Data clusters (Group J) - NEW"),
        mpatches.Patch(facecolor="#aaaaaa",         label="Model 2 features (Groups A-G)"),
    ], loc="lower right", fontsize=10)
    for bar, feat, imp in zip(bars, df["feature"], df["importance"]):
        clr = _classify(feat)
        if clr != "#aaaaaa":
            ax.text(imp + 0.0001, bar.get_y() + bar.get_height()/2,
                    f"  * {imp:.4f}", va="center", fontsize=8, color=clr, fontweight="bold")
    _save("new_features_importance.png")


def plot_yearly_m6_vs_m2(wf: pd.DataFrame, acc_m2_holdout: float):
    """Walk-forward showing M6 vs logistic with M2 holdout as reference line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m6, "o-", color="darkgreen", lw=2,
            label="Model 6 Final - Elo + Serve + Playstyle")
    ax.plot(wf.test_year, wf.acc_lr,  "s--", color="steelblue", lw=1.5,
            label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.2,
            label="Rank baseline")
    ax.axhline(acc_m2_holdout, color="darkorange", linestyle="-.", lw=1.5, alpha=0.8,
               label=f"Model 2 holdout accuracy ({acc_m2_holdout:.4f})")
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m6,
                    alpha=0.12, color="darkgreen", label="M6 gain over logistic")
    ax.set_xlabel("Test Year"); ax.set_ylabel("Accuracy")
    ax.set_title("Walk-Forward Accuracy - Model 6 Final vs Model 2 Reference\n"
                 "(orange dashed = Model 2 holdout accuracy)")
    ax.legend(fontsize=9); ax.set_ylim(0.55, 0.80); ax.grid(axis="y", alpha=0.3)
    for _, r in wf.iterrows():
        delta = r.acc_m6 - r.acc_lr
        ax.annotate(f"+{delta:.3f}", xy=(r.test_year, (r.acc_m6+r.acc_lr)/2),
                    ha="center", fontsize=7.5, color="darkgreen", fontweight="bold")
    _save("m6_vs_m2_yearly.png")


def plot_cluster_profiles(centers: pd.DataFrame):
    """
    Radar-style bar chart showing the stat profile of each data-driven cluster.
    Makes the clusters interpretable: which ones are big servers, returners, etc.
    """
    stats = ["ace_rate", "df_rate", "first_in_rate", "first_win_rate",
             "second_win_rate", "bp_save_rate", "return_win_rate"]
    labels = ["Ace\nRate", "DF\nRate", "1st In\nRate", "1st Srv\nWin%",
              "2nd Srv\nWin%", "BP\nSave%", "Return\nWin%"]

    n = len(centers)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    x = np.arange(len(stats)); w = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (_, row) in enumerate(centers.iterrows()):
        vals = [row.get(s, 0) for s in stats]
        offset = (i - n/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=colors[i], alpha=0.85,
                      label=f"{row['cluster']} (n={int(row.get('size',0)):,})")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Rate (0-1)")
    ax.set_title("Data-Driven Player Cluster Profiles (KMeans, k=6)\n"
                 "Each cluster represents a distinct serve/return style archetype")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _save("cluster_profiles.png")


# ==============================================================================
# 9.  MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 70)
    print("  Model 6 - Final: XGBoost + Elo + Serve Stats + Playstyle + Clusters")
    print("=" * 70)

    print("\nLoading data ...")
    processed    = load_processed(PROC_DIR)
    matches      = load_initial_matches(INIT_DIR)
    surf_df      = load_surface_results(INIT_DIR)
    serve_df     = load_serve_data(SERVE_DIR)
    style_lookup = load_playstyle_data(PLAYSTYLE_DIR)

    print("\nBuilding static lookups ...")
    player_attrs    = build_player_attrs(matches)
    surf_lookup     = build_surface_lookup(surf_df)
    serve_lookup    = build_serve_lookup(serve_df)
    cluster_lookup, cluster_centers = build_player_clusters(matches)

    print("\nComputing dynamic features (Elo, form, H2H) ...")
    processed = compute_dynamic_features(processed)

    print("\nBuilding symmetric dataset ...")
    data = build_dataset(processed, player_attrs, surf_lookup, serve_lookup,
                         style_lookup, cluster_lookup)

    data_comp = data[(data["year"] >= 2014) & (data["year"] <= 2024)].copy()

    # -- Walk-forward 2016-2024 ------------------------------------------------
    print("\nRunning walk-forward validation (2016-2024) ...")
    wf = walk_forward(data_comp)

    # -- Multi-holdout anti-overfitting check ----------------------------------
    print("\nRunning multi-holdout validation (anti-overfitting check) ...")
    holdout_results = multi_holdout_validation(data_comp)

    # -- Final holdout: train 2014-2020 / test 2021-2024 ----------------------
    print("\nTraining final models (train 2014-2020 / test 2021-2024) ...")
    train = data_comp[data_comp["year"] <= 2020]
    test  = data_comp[data_comp["year"] >= 2021]

    final_model = make_m6_pipeline()
    acc6, auc6, brier6, probs6 = fit_eval(
        final_model,
        train[NUMERIC_COLS + CAT_COLS], train["label"],
        test[NUMERIC_COLS + CAT_COLS],  test["label"],
    )

    print("  Training Model 2 equivalent for comparison ...")
    m2_model = make_m2_pipeline()
    acc2, auc2, brier2, probs2 = fit_eval(
        m2_model,
        train[NUMERIC_M2 + CAT_M2], train["label"],
        test[NUMERIC_M2 + CAT_M2],  test["label"],
    )

    lr_model = make_baseline_pipeline()
    acc_lr, auc_lr, brier_lr, probs_lr = fit_eval(
        lr_model,
        train[BASELINE_NUM + BASELINE_CAT], train["label"],
        test[BASELINE_NUM + BASELINE_CAT],  test["label"],
    )

    rank_base = accuracy_score(test["label"], (test["rank_diff"] < 0).astype(int))

    # -- Print results ---------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  Results - Model 6 Final vs Model 2 vs Logistic Baseline vs Rank Baseline")
    print(f"{'='*72}")
    print(f"{'Metric':<28} {'Model 6':>12}  {'Model 2':>10}  {'Logistic LR':>12}  {'Rank base':>10}")
    print(f"{'-'*72}")
    print(f"{'Accuracy':<28} {acc6:>12.4f}  {acc2:>10.4f}  {acc_lr:>12.4f}  {rank_base:>10.4f}")
    print(f"{'ROC AUC':<28} {auc6:>12.4f}  {auc2:>10.4f}  {auc_lr:>12.4f}  {'-':>10}")
    print(f"{'Brier Score':<28} {brier6:>12.4f}  {brier2:>10.4f}  {brier_lr:>12.4f}  {'-':>10}")
    print(f"{'-'*72}")
    print(f"{'D Accuracy  M6 vs M2':<28} {acc6 - acc2:>+12.4f}")
    print(f"{'D AUC       M6 vs M2':<28} {auc6 - auc2:>+12.4f}")
    print(f"{'D Brier     M6 vs M2':<28} {brier6 - brier2:>+12.4f}  (negative = better)")
    print(f"{'-'*72}")
    print(f"{'D Accuracy  M6 vs LR':<28} {acc6 - acc_lr:>+12.4f}")
    print(f"{'D AUC       M6 vs LR':<28} {auc6 - auc_lr:>+12.4f}")
    print(f"{'='*72}\n")

    print("Multi-holdout summary:")
    for r in holdout_results:
        if not np.isnan(r.get("acc_m6", np.nan)):
            print(f"  {r['name']}: M6={r['acc_m6']:.4f}  M2={r['acc_m2']:.4f}  "
                  f"D={r['acc_m6']-r['acc_m2']:+.4f}")

    joblib.dump(final_model, os.path.join(ROOT, "model_6.joblib"))
    print(f"\nSaved model -> {ROOT}/model_6.joblib")

    # -- Generate all plots ----------------------------------------------------
    print("\nGenerating plots ...")

    # Standard diagnostic plots
    plot_roc_curve(test["label"], probs6, acc6, auc6)
    plot_calibration(test["label"], probs6)
    plot_confusion_matrix(test["label"], probs6)
    plot_feature_importance(final_model)
    plot_walk_forward(wf)
    plot_accuracy_by_surface(test, final_model)
    plot_accuracy_by_confidence(test["label"], probs6)
    plot_prob_histogram(test["label"], probs6)
    plot_vs_baseline_roc(test["label"], probs6, probs_lr)
    plot_yearly_comparison(wf)
    plot_feature_groups(final_model)
    plot_elo_validation(test)
    plot_accuracy_by_round(test, final_model)
    plot_rank_gap_accuracy(test, probs6)

    # New Model 6 specific plots
    plot_multi_holdout(holdout_results)
    plot_m6_vs_all_comparison(acc6, auc6, brier6, acc2, auc2, brier2,
                               acc_lr, auc_lr, brier_lr)
    plot_new_features_importance(final_model)
    plot_yearly_m6_vs_m2(wf, acc2)
    plot_cluster_profiles(cluster_centers)

    print(f"\nAll 19 plots saved to: {PLOTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
