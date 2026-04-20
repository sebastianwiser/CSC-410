#!/usr/bin/env python3
"""
model_4.py - Model 2 + Injury Reports (Impact Analysis)
=======================================================
Builds directly on Model 2 by adding pre-match player injury information from
CSC410Pres/data/injury_reports/. The goal is to test whether injury features
improve predictions and to produce tennis-specific charts that tell a clear
story about recent injuries, returns, and match outcomes.

Data sources:
  data/processed/atp_tennis.csv
  data/initial/atp_matches_YYYY.csv
  data/initial/player_surface_results.csv
  data/injury_reports/*.json

Validation:
  1. Expanding walk-forward - test years 2016-2024
  2. Single holdout         - train 2014-2020, test 2021-2024

Usage:
  cd CSC410Pres/models/model_4
  python model_4.py
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "..", "data")
INIT_DIR = os.path.join(DATA_DIR, "initial")
PROC_DIR = os.path.join(DATA_DIR, "processed")
INJURY_DIR = os.path.join(DATA_DIR, "injury_reports")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ELO_BASE = 1500.0
ELO_K = 32.0
ELO_K_SURF = 24.0
FORM_SHORT = 5
FORM_LONG = 20

INJURY_LOOKBACK_DAYS = 365
INJURY_RECENT_RETURN_DAYS = 90
INJURY_NO_EVENT_DAYS = 730

# Model 2 base features retained for direct comparison.
NUMERIC_COLS_M2 = [
    "rank_diff",
    "pts_diff",
    "rank_ratio",
    "log_rank_diff",
    "elo_diff",
    "elo_surf_diff",
    "surface_specialist_diff",
    "form_short_diff",
    "form_long_diff",
    "surf_form_diff",
    "form_trend_diff",
    "h2h_rate",
    "h2h_n",
    "days_rest_diff",
    "experience_diff",
    "age_diff",
    "ht_diff",
    "surf_win_rate_diff",
    "surf_exp_diff",
]
CAT_COLS = ["surface", "series", "court", "round", "best_of", "p1_hand", "p2_hand"]

INJURY_NUMERIC = [
    "inj_recent_flag_diff",
    "inj_count_365_diff",
    "inj_days_since_last_diff",
    "inj_joint_recent_diff",
    "inj_chronic_recent_diff",
    "inj_retired_recent_diff",
    "inj_recovery_days_diff",
    "inj_recent_return_diff",
]
NUMERIC_COLS_M4 = NUMERIC_COLS_M2 + INJURY_NUMERIC
NUMERIC_COLS = NUMERIC_COLS_M4

BASELINE_NUM = ["rank_diff", "log_rank_diff", "rank_ratio", "pts_diff", "age_diff", "h2h_rate"]
BASELINE_CAT = ["surface", "series", "round", "best_of"]

GROUP_COLORS = [
    ("num__elo", "darkorange"),
    ("num__surf_form", "steelblue"),
    ("num__surface_specialist", "darkorange"),
    ("num__form", "steelblue"),
    ("num__h2h", "seagreen"),
    ("num__rank", "mediumpurple"),
    ("num__pts", "mediumpurple"),
    ("num__log", "mediumpurple"),
    ("num__days", "dimgray"),
    ("num__exp", "dimgray"),
    ("num__age", "orchid"),
    ("num__ht", "orchid"),
    ("num__surf_win", "teal"),
    ("num__surf_exp", "teal"),
    ("num__inj_", "crimson"),
]

GROUP_LABELS = {
    "A - Rank/skill": ["rank_diff", "pts_diff", "rank_ratio", "log_rank_diff"],
    "B - Elo": ["elo_diff", "elo_surf_diff", "surface_specialist_diff"],
    "C - Rolling form": ["form_short_diff", "form_long_diff", "surf_form_diff", "form_trend_diff"],
    "D - H2H": ["h2h_rate", "h2h_n"],
    "E - Rest/experience": ["days_rest_diff", "experience_diff"],
    "F - Player attributes": ["age_diff", "ht_diff"],
    "G - Surface history": ["surf_win_rate_diff", "surf_exp_diff"],
    "H - Injury reports": [
        "inj_recent_flag_diff",
        "inj_count_365_diff",
        "inj_days_since_last_diff",
        "inj_joint_recent_diff",
        "inj_chronic_recent_diff",
        "inj_retired_recent_diff",
        "inj_recovery_days_diff",
        "inj_recent_return_diff",
    ],
}

GROUP_COLORS_BAR = {
    "A - Rank/skill": "mediumpurple",
    "B - Elo": "darkorange",
    "C - Rolling form": "steelblue",
    "D - H2H": "seagreen",
    "E - Rest/experience": "dimgray",
    "F - Player attributes": "orchid",
    "G - Surface history": "teal",
    "H - Injury reports": "crimson",
}


def _norm(name: str) -> str:
    text = str(name).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_atp(full_name: str) -> str:
    parts = str(full_name).strip().split()
    if len(parts) >= 2:
        last_name = " ".join(parts[1:])
        first_initial = parts[0][0]
        return _norm(f"{last_name} {first_initial}.")
    return _norm(full_name)


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _form(dq: deque, n: int | None = None) -> float:
    values = list(dq)[-n:] if n else list(dq)
    return float(np.mean(values)) if values else 0.5


def _feat_color(feat: str) -> str:
    for prefix, color in GROUP_COLORS:
        if feat.startswith(prefix):
            return color
    return "silver"


def _save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=220, bbox_inches="tight")
    plt.close()


def _extract_importances(model) -> tuple[np.ndarray, np.ndarray] | None:
    pipe = None
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        pipe = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "named_steps"):
        pipe = model
    if pipe is None or not hasattr(pipe, "named_steps"):
        return None
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    return pre.get_feature_names_out(), clf.feature_importances_


def _parse_injury_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _parse_recovery_days(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "unknown":
        return None
    text = text.replace("about ", "").strip()

    def _to_days(amount: float, unit: str) -> float:
        if "month" in unit:
            return amount * 30.0
        if "week" in unit:
            return amount * 7.0
        return amount

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", text)
    if match:
        lo = float(match.group(1))
        hi = float(match.group(2))
        return _to_days((lo + hi) / 2.0, match.group(3))

    match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", text)
    if match:
        return _to_days(float(match.group(1)), match.group(2))

    if "same day" in text:
        return 2.0
    if "day" in text and "week" in text:
        return 7.0
    if "day" in text:
        return 2.0
    if "week" in text:
        return 14.0
    return None


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
    print(f"  Processed: {len(df):,} matches ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_initial_matches(init_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(init_dir, "atp_matches_*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv in {init_dir}")
    df = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df["year"] = df["match_date"].dt.year
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "winner_ht", "loser_ht"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  Initial: {len(df):,} matches ({int(df['year'].min())}-{int(df['year'].max())})")
    return df


def load_surface_results(init_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(init_dir, "player_surface_results.csv"))
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["surface"] = df["surface"].astype(str).str.strip().str.lower()
    for col in ["wins", "losses", "matches"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "matches" not in df.columns:
        df["matches"] = df.get("wins", 0) + df.get("losses", 0)
    df["win_rate"] = np.where(
        df["matches"] > 0,
        df.get("wins", df["matches"] * 0.5) / df["matches"],
        0.5,
    )
    df["name_key"] = df["player_name"].map(_norm)
    print(f"  Surface results: {len(df):,} rows ({df['name_key'].nunique():,} unique players)")
    return df


def load_injury_data(injury_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(injury_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No injury report JSON files in {injury_dir}")

    profiles: dict[str, dict] = {}
    total_events = 0
    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        for player_name, pdata in raw.get("players", {}).items():
            canonical = _norm(player_name)
            profile = profiles.setdefault(
                canonical,
                {"aliases": set(), "events": {}},
            )
            profile["aliases"].update([canonical, _norm_atp(player_name)])
            for injury in pdata.get("injuries", []):
                injury_date = _parse_injury_date(injury.get("injury_date"))
                if injury_date is None:
                    continue
                recovery_days = _parse_recovery_days(injury.get("estimated_recovery_time"))
                retired = str(injury.get("retired_during_match_due_to_injury", "no")).strip().lower() in {
                    "yes",
                    "true",
                    "1",
                }
                chronic_label = str(injury.get("acute_or_chronic", "unknown")).strip().lower()
                event_key = (
                    str(injury_date.date()),
                    _norm(injury.get("injury_name", "")),
                    bool(injury.get("joint_injury", False)),
                    chronic_label,
                    retired,
                    None if recovery_days is None else round(float(recovery_days), 2),
                )
                if event_key in profile["events"]:
                    continue
                profile["events"][event_key] = {
                    "date": injury_date,
                    "recovery_days": recovery_days,
                    "return_date": (
                        injury_date + pd.Timedelta(days=float(recovery_days))
                        if recovery_days is not None
                        else None
                    ),
                    "joint": bool(injury.get("joint_injury", False)),
                    "chronic": chronic_label == "chronic",
                    "acute": chronic_label == "acute",
                    "retired": retired,
                }
                total_events += 1

    lookup: dict[str, dict] = {}
    for profile in profiles.values():
        events = sorted(profile["events"].values(), key=lambda item: item["date"])
        out = {"has_report": 1, "events": events}
        for alias in profile["aliases"]:
            if alias:
                lookup[alias] = out

    print(f"  Injury reports: {len(profiles):,} players, {total_events:,} unique events from {len(files)} files")
    return lookup


def build_player_attrs(matches: pd.DataFrame) -> dict:
    winner = pd.DataFrame(
        {
            "name_key": matches["winner_name"].map(_norm),
            "hand": matches.get("winner_hand", pd.Series("U", index=matches.index)),
            "ht": matches.get("winner_ht", pd.Series(np.nan, index=matches.index)),
            "birth_year": matches["year"] - matches["winner_age"],
        }
    )
    loser = pd.DataFrame(
        {
            "name_key": matches["loser_name"].map(_norm),
            "hand": matches.get("loser_hand", pd.Series("U", index=matches.index)),
            "ht": matches.get("loser_ht", pd.Series(np.nan, index=matches.index)),
            "birth_year": matches["year"] - matches["loser_age"],
        }
    )
    df = pd.concat([winner, loser], ignore_index=True)
    df = df[df["name_key"] != ""]

    def _mode(series: pd.Series) -> str:
        mode = series.dropna().mode()
        return str(mode.iloc[0]) if len(mode) else "U"

    result: dict[str, dict] = {}
    for key, grp in df.groupby("name_key"):
        result[key] = {
            "hand": _mode(grp["hand"]),
            "ht": float(grp["ht"].median()) if grp["ht"].notna().any() else np.nan,
            "birth_year": float(grp["birth_year"].median()) if grp["birth_year"].notna().any() else np.nan,
        }
    print(f"  Player attrs: {len(result):,} players indexed")
    return result


def build_surface_lookup(surf_df: pd.DataFrame) -> dict:
    lookup: dict[tuple, dict] = {}
    for _, row in surf_df.iterrows():
        lookup[(row["name_key"], row["surface"])] = {
            "win_rate": float(row["win_rate"]),
            "n_matches": int(row["matches"]),
        }
    print(f"  Surface lookup: {len(lookup):,} (player, surface) entries")
    return lookup


def _player_injury_snapshot(player: str, match_dt: pd.Timestamp, injury_lookup: dict) -> dict:
    profile = injury_lookup.get(_norm(player))
    neutral = {
        "has_report": 0,
        "recent_flag": 0.0,
        "count_365": 0.0,
        "days_since_last": float(INJURY_NO_EVENT_DAYS),
        "joint_recent": 0.0,
        "chronic_recent": 0.0,
        "retired_recent": 0.0,
        "recovery_days_recent": np.nan,
        "recent_return": 0.0,
    }
    if profile is None:
        return neutral

    events = [event for event in profile["events"] if event["date"] < match_dt]
    if not events:
        neutral["has_report"] = 1
        return neutral

    recent = [
        event for event in events
        if 0 <= (match_dt - event["date"]).days <= INJURY_LOOKBACK_DAYS
    ]
    recent_with_recovery = [event for event in recent if event["recovery_days"] is not None]
    last_event = events[-1]
    recent_return = any(
        event["return_date"] is not None and 0 <= (match_dt - event["return_date"]).days <= INJURY_RECENT_RETURN_DAYS
        for event in events
    )

    return {
        "has_report": 1,
        "recent_flag": float(bool(recent)),
        "count_365": float(len(recent)),
        "days_since_last": float(min((match_dt - last_event["date"]).days, INJURY_NO_EVENT_DAYS)),
        "joint_recent": float(any(event["joint"] for event in recent)),
        "chronic_recent": float(any(event["chronic"] for event in recent)),
        "retired_recent": float(any(event["retired"] for event in recent)),
        "recovery_days_recent": (
            float(recent_with_recovery[-1]["recovery_days"]) if recent_with_recovery else np.nan
        ),
        "recent_return": float(recent_return),
    }


def compute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    elo_all = defaultdict(lambda: ELO_BASE)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_BASE))
    res_all = defaultdict(lambda: deque(maxlen=FORM_LONG))
    res_surf = defaultdict(lambda: defaultdict(lambda: deque(maxlen=FORM_LONG)))
    last_dt: dict = {}
    exp_cnt = defaultdict(int)
    h2h_wins = defaultdict(int)

    feat_rows = []
    for row in df.itertuples(index=False):
        p1 = getattr(row, "Player_1", None)
        p2 = getattr(row, "Player_2", None)
        if not isinstance(p1, str) or not isinstance(p2, str):
            feat_rows.append({})
            continue

        win = getattr(row, "Winner", None)
        surf = str(getattr(row, "Surface", "unknown") or "unknown").lower()
        date = getattr(row, "Date", None)
        p1_won = win == p1

        p1_elo = elo_all[p1]
        p2_elo = elo_all[p2]
        p1_elo_s = elo_surf[p1][surf]
        p2_elo_s = elo_surf[p2][surf]

        p1_fs = _form(res_all[p1], FORM_SHORT)
        p2_fs = _form(res_all[p2], FORM_SHORT)
        p1_fl = _form(res_all[p1])
        p2_fl = _form(res_all[p2])
        p1_sf = _form(res_surf[p1][surf])
        p2_sf = _form(res_surf[p2][surf])

        h2h_p1 = h2h_wins[(p1, p2)]
        h2h_p2 = h2h_wins[(p2, p1)]
        h2h_total = h2h_p1 + h2h_p2
        h2h_rate = (h2h_p1 + 1.0) / (h2h_total + 2.0)

        p1_days = float((date - last_dt[p1]).days) if p1 in last_dt else 30.0
        p2_days = float((date - last_dt[p2]).days) if p2 in last_dt else 30.0

        feat_rows.append(
            {
                "p1_elo_pre": p1_elo,
                "p2_elo_pre": p2_elo,
                "p1_eloS_pre": p1_elo_s,
                "p2_eloS_pre": p2_elo_s,
                "p1_fS_pre": p1_fs,
                "p2_fS_pre": p2_fs,
                "p1_fL_pre": p1_fl,
                "p2_fL_pre": p2_fl,
                "p1_sf_pre": p1_sf,
                "p2_sf_pre": p2_sf,
                "p1_h2h_rate_pre": h2h_rate,
                "h2h_n_pre": float(h2h_total),
                "p1_days_pre": p1_days,
                "p2_days_pre": p2_days,
                "p1_exp_pre": float(exp_cnt[p1]),
                "p2_exp_pre": float(exp_cnt[p2]),
            }
        )

        score = 1.0 if p1_won else 0.0
        expected = _elo_expected(p1_elo, p2_elo)
        elo_all[p1] += ELO_K * (score - expected)
        elo_all[p2] += ELO_K * ((1 - score) - (1 - expected))

        expected_s = _elo_expected(p1_elo_s, p2_elo_s)
        elo_surf[p1][surf] += ELO_K_SURF * (score - expected_s)
        elo_surf[p2][surf] += ELO_K_SURF * ((1 - score) - (1 - expected_s))

        res_all[p1].append(1.0 if p1_won else 0.0)
        res_all[p2].append(0.0 if p1_won else 1.0)
        res_surf[p1][surf].append(1.0 if p1_won else 0.0)
        res_surf[p2][surf].append(0.0 if p1_won else 1.0)

        if p1_won:
            h2h_wins[(p1, p2)] += 1
        else:
            h2h_wins[(p2, p1)] += 1

        last_dt[p1] = date
        last_dt[p2] = date
        exp_cnt[p1] += 1
        exp_cnt[p2] += 1

    return pd.concat([df, pd.DataFrame(feat_rows, index=df.index)], axis=1)


def _make_rows(df: pd.DataFrame, flip: bool, player_attrs: dict, surf_lookup: dict, injury_lookup: dict) -> pd.DataFrame:
    def _v(a: str, b: str):
        col = b if flip else a
        return df[col].values if col in df.columns else np.full(len(df), np.nan)

    p1_names = df["Player_2" if flip else "Player_1"].values
    p2_names = df["Player_1" if flip else "Player_2"].values
    label = (1 - df["p1_wins"]).values if flip else df["p1_wins"].values
    h2h_rate = (
        1.0 - df["p1_h2h_rate_pre"].values if flip else df["p1_h2h_rate_pre"].values
    ) if "p1_h2h_rate_pre" in df.columns else np.full(len(df), 0.5)

    surf_vals = df["Surface"].values
    match_years = df["year"].values.astype(float)
    match_dates = pd.to_datetime(df["Date"]).tolist()

    def _flt(names, key):
        return np.array(
            [float(player_attrs.get(_norm(name), {}).get(key, np.nan)) for name in names],
            dtype=float,
        )

    def _srf(names, key, default):
        return np.array(
            [
                float(surf_lookup.get((_norm(name), str(surface or "").lower()), {}).get(key, default))
                for name, surface in zip(names, surf_vals)
            ],
            dtype=float,
        )

    p1_birth_year = _flt(p1_names, "birth_year")
    p2_birth_year = _flt(p2_names, "birth_year")
    p1_height = _flt(p1_names, "ht")
    p2_height = _flt(p2_names, "ht")
    p1_hand = np.array([player_attrs.get(_norm(name), {}).get("hand", "U") for name in p1_names])
    p2_hand = np.array([player_attrs.get(_norm(name), {}).get("hand", "U") for name in p2_names])

    p1_snapshots = [_player_injury_snapshot(name, dt, injury_lookup) for name, dt in zip(p1_names, match_dates)]
    p2_snapshots = [_player_injury_snapshot(name, dt, injury_lookup) for name, dt in zip(p2_names, match_dates)]

    def _snap(snapshots, key):
        return np.array([snap[key] for snap in snapshots], dtype=float)

    return pd.DataFrame(
        {
            "year": df["year"].values,
            "p1_rank": _v("Rank_1", "Rank_2"),
            "p2_rank": _v("Rank_2", "Rank_1"),
            "p1_pts": _v("Pts_1", "Pts_2"),
            "p2_pts": _v("Pts_2", "Pts_1"),
            "p1_elo": _v("p1_elo_pre", "p2_elo_pre"),
            "p2_elo": _v("p2_elo_pre", "p1_elo_pre"),
            "p1_eloS": _v("p1_eloS_pre", "p2_eloS_pre"),
            "p2_eloS": _v("p2_eloS_pre", "p1_eloS_pre"),
            "p1_fS": _v("p1_fS_pre", "p2_fS_pre"),
            "p2_fS": _v("p2_fS_pre", "p1_fS_pre"),
            "p1_fL": _v("p1_fL_pre", "p2_fL_pre"),
            "p2_fL": _v("p2_fL_pre", "p1_fL_pre"),
            "p1_sf": _v("p1_sf_pre", "p2_sf_pre"),
            "p2_sf": _v("p2_sf_pre", "p1_sf_pre"),
            "h2h_rate": h2h_rate,
            "h2h_n": df["h2h_n_pre"].values if "h2h_n_pre" in df.columns else np.zeros(len(df)),
            "p1_days": _v("p1_days_pre", "p2_days_pre"),
            "p2_days": _v("p2_days_pre", "p1_days_pre"),
            "p1_exp": _v("p1_exp_pre", "p2_exp_pre"),
            "p2_exp": _v("p2_exp_pre", "p1_exp_pre"),
            "p1_age": match_years - p1_birth_year,
            "p2_age": match_years - p2_birth_year,
            "p1_ht": p1_height,
            "p2_ht": p2_height,
            "p1_hand": p1_hand,
            "p2_hand": p2_hand,
            "p1_swr": _srf(p1_names, "win_rate", np.nan),
            "p2_swr": _srf(p2_names, "win_rate", np.nan),
            "p1_sn": _srf(p1_names, "n_matches", 0),
            "p2_sn": _srf(p2_names, "n_matches", 0),
            "p1_has_injury_report": _snap(p1_snapshots, "has_report"),
            "p2_has_injury_report": _snap(p2_snapshots, "has_report"),
            "p1_inj_recent_flag": _snap(p1_snapshots, "recent_flag"),
            "p2_inj_recent_flag": _snap(p2_snapshots, "recent_flag"),
            "p1_inj_count_365": _snap(p1_snapshots, "count_365"),
            "p2_inj_count_365": _snap(p2_snapshots, "count_365"),
            "p1_inj_days_since_last": _snap(p1_snapshots, "days_since_last"),
            "p2_inj_days_since_last": _snap(p2_snapshots, "days_since_last"),
            "p1_inj_joint_recent": _snap(p1_snapshots, "joint_recent"),
            "p2_inj_joint_recent": _snap(p2_snapshots, "joint_recent"),
            "p1_inj_chronic_recent": _snap(p1_snapshots, "chronic_recent"),
            "p2_inj_chronic_recent": _snap(p2_snapshots, "chronic_recent"),
            "p1_inj_retired_recent": _snap(p1_snapshots, "retired_recent"),
            "p2_inj_retired_recent": _snap(p2_snapshots, "retired_recent"),
            "p1_inj_recovery_days_recent": _snap(p1_snapshots, "recovery_days_recent"),
            "p2_inj_recovery_days_recent": _snap(p2_snapshots, "recovery_days_recent"),
            "p1_inj_recent_return": _snap(p1_snapshots, "recent_return"),
            "p2_inj_recent_return": _snap(p2_snapshots, "recent_return"),
            "surface": df["Surface"].values,
            "series": df["Series"].values if "Series" in df.columns else np.full(len(df), "unknown"),
            "court": df["Court"].values if "Court" in df.columns else np.full(len(df), "unknown"),
            "round": df["Round"].values,
            "best_of": df["best_of"].values,
            "label": label,
        }
    )


def build_dataset(df: pd.DataFrame, player_attrs: dict, surf_lookup: dict, injury_lookup: dict) -> pd.DataFrame:
    df = df.copy()
    df["p1_wins"] = (df["Player_1"] == df["Winner"]).astype(int)
    data = pd.concat(
        [
            _make_rows(df, False, player_attrs, surf_lookup, injury_lookup),
            _make_rows(df, True, player_attrs, surf_lookup, injury_lookup),
        ],
        ignore_index=True,
    )

    data["rank_diff"] = data["p1_rank"] - data["p2_rank"]
    data["pts_diff"] = data["p1_pts"].fillna(0) - data["p2_pts"].fillna(0)
    data["rank_ratio"] = data["p1_rank"] / data["p2_rank"].clip(lower=1)
    data["log_rank_diff"] = np.log1p(data["p1_rank"]) - np.log1p(data["p2_rank"])

    data["elo_diff"] = data["p1_elo"] - data["p2_elo"]
    data["elo_surf_diff"] = data["p1_eloS"] - data["p2_eloS"]
    data["surface_specialist_diff"] = data["elo_surf_diff"] - data["elo_diff"]

    data["form_short_diff"] = data["p1_fS"] - data["p2_fS"]
    data["form_long_diff"] = data["p1_fL"] - data["p2_fL"]
    data["surf_form_diff"] = data["p1_sf"] - data["p2_sf"]
    data["form_trend_diff"] = data["form_short_diff"] - data["form_long_diff"]

    data["days_rest_diff"] = data["p1_days"] - data["p2_days"]
    data["experience_diff"] = data["p1_exp"] - data["p2_exp"]

    data["age_diff"] = data["p1_age"] - data["p2_age"]
    data["ht_diff"] = data["p1_ht"] - data["p2_ht"]

    data["surf_win_rate_diff"] = data["p1_swr"] - data["p2_swr"]
    data["surf_exp_diff"] = data["p1_sn"] - data["p2_sn"]

    data["inj_recent_flag_diff"] = data["p1_inj_recent_flag"] - data["p2_inj_recent_flag"]
    data["inj_count_365_diff"] = data["p1_inj_count_365"] - data["p2_inj_count_365"]
    data["inj_days_since_last_diff"] = data["p1_inj_days_since_last"] - data["p2_inj_days_since_last"]
    data["inj_joint_recent_diff"] = data["p1_inj_joint_recent"] - data["p2_inj_joint_recent"]
    data["inj_chronic_recent_diff"] = data["p1_inj_chronic_recent"] - data["p2_inj_chronic_recent"]
    data["inj_retired_recent_diff"] = data["p1_inj_retired_recent"] - data["p2_inj_retired_recent"]
    data["inj_recovery_days_diff"] = (
        data["p1_inj_recovery_days_recent"] - data["p2_inj_recovery_days_recent"]
    )
    data["inj_recent_return_diff"] = data["p1_inj_recent_return"] - data["p2_inj_recent_return"]

    data["injury_coverage"] = data["p1_has_injury_report"] + data["p2_has_injury_report"]

    print(f"  Dataset: {len(data):,} rows ({int(data['year'].min())}-{int(data['year'].max())})")
    coverage = data["injury_coverage"].value_counts().sort_index()
    for level, count in coverage.items():
        label = {0: "Neither player", 1: "One player", 2: "Both players"}.get(int(level), str(level))
        print(f"    Injury coverage - {label}: {count:,} rows ({count / len(data) * 100:.1f}%)")
    return data


def _num_cat_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def make_m4_pipeline() -> CalibratedClassifierCV:
    pre = _num_cat_preprocessor(NUMERIC_COLS_M4, CAT_COLS)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=15,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]), cv=3, method="isotonic")


def make_m2_pipeline() -> CalibratedClassifierCV:
    pre = _num_cat_preprocessor(NUMERIC_COLS_M2, CAT_COLS)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=15,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return CalibratedClassifierCV(Pipeline([("pre", pre), ("clf", xgb)]), cv=3, method="isotonic")


def make_baseline_pipeline() -> Pipeline:
    pre = _num_cat_preprocessor(BASELINE_NUM, BASELINE_CAT)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return Pipeline([("pre", pre), ("clf", lr)])


def fit_eval(model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    probs = model.predict_proba(x_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return (
        accuracy_score(y_te, preds),
        roc_auc_score(y_te, probs),
        brier_score_loss(y_te, probs),
        probs,
    )


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_year in sorted(data["year"].unique()):
        if test_year < 2016 or test_year > 2024:
            continue
        train = data[data["year"] < test_year]
        test = data[data["year"] == test_year]
        if len(train) < 100 or len(test) < 50:
            continue

        y_train = train["label"]
        y_test = test["label"]

        m4 = make_m4_pipeline()
        acc_m4, auc_m4, _, _ = fit_eval(
            m4,
            train[NUMERIC_COLS_M4 + CAT_COLS],
            y_train,
            test[NUMERIC_COLS_M4 + CAT_COLS],
            y_test,
        )

        lr = make_baseline_pipeline()
        acc_lr, auc_lr, _, _ = fit_eval(
            lr,
            train[BASELINE_NUM + BASELINE_CAT],
            y_train,
            test[BASELINE_NUM + BASELINE_CAT],
            y_test,
        )

        rank_base = accuracy_score(y_test, (test["rank_diff"] < 0).astype(int))
        rows.append(
            {
                "test_year": test_year,
                "acc_m4": acc_m4,
                "auc_m4": auc_m4,
                "acc_lr": acc_lr,
                "auc_lr": auc_lr,
                "rank_baseline": rank_base,
            }
        )

    wf = pd.DataFrame(rows)
    if not wf.empty:
        print("\nWalk-forward (2016-2024)")
        print(wf.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return wf


def plot_roc_curve(y_true, probs, acc: float, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"Model 4 XGBoost+Injuries (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Model 4\n(Test 2021-2024 | Accuracy={acc:.4f})")
    ax.legend()
    _save("roc_curve.png")


def plot_calibration(y_true, probs) -> None:
    fop, mpv = calibration_curve(y_true, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="darkorange", lw=2, label="Model 4")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve - Model 4")
    ax.legend()
    _save("calibration.png")


def plot_confusion_matrix_chart(y_true, probs) -> None:
    cm = confusion_matrix(y_true, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Loss", "Predicted Win"])
    ax.set_yticklabels(["Actual Loss", "Actual Win"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix - Model 4 (Test 2021-2024)")
    plt.colorbar(im, ax=ax)
    _save("confusion_matrix.png")


def plot_feature_importance(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    df.to_csv(os.path.join(PLOTS_DIR, "feature_importance.csv"), index=False)
    top = df.head(25).sort_values("importance", ascending=True)
    colors = [_feat_color(feature) for feature in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Top 25 Feature Importances - Model 4 (crimson = injury features)")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="crimson", label="Injury reports (Group H)"),
            mpatches.Patch(facecolor="darkorange", label="Elo"),
            mpatches.Patch(facecolor="steelblue", label="Rolling form"),
            mpatches.Patch(facecolor="teal", label="Surface history"),
            mpatches.Patch(facecolor="mediumpurple", label="Rank/skill"),
        ],
        fontsize=9,
    )
    _save("feature_importance.png")


def plot_walk_forward(wf: pd.DataFrame) -> None:
    if wf.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.plot(wf.test_year, wf.acc_m4, "o-", color="darkorange", lw=2, label="Model 4")
    ax1.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.5, label="Rank baseline")
    ax1.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Walk-Forward Validation - Accuracy (Model 4)")
    ax1.legend()
    ax1.set_ylim(0.55, 0.82)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(wf.test_year, wf.auc_m4, "o-", color="darkorange", lw=2, label="Model 4")
    ax2.plot(wf.test_year, wf.auc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax2.set_xlabel("Test Year")
    ax2.set_ylabel("ROC AUC")
    ax2.set_title("Walk-Forward Validation - ROC AUC (Model 4)")
    ax2.set_ylim(0.60, 0.85)
    ax2.grid(axis="y", alpha=0.3)
    _save("walk_forward.png")


def plot_accuracy_by_surface(test: pd.DataFrame, model) -> None:
    valid, accs, bases, ns = [], [], [], []
    for surf in sorted(s for s in test["surface"].unique() if s not in ("unknown", "nan") and pd.notna(s)):
        subset = test[test["surface"] == surf]
        if len(subset) < 100:
            continue
        probs = model.predict_proba(subset[NUMERIC_COLS_M4 + CAT_COLS])[:, 1]
        accs.append(accuracy_score(subset["label"], (probs >= 0.5).astype(int)))
        bases.append(accuracy_score(subset["label"], (subset["rank_diff"] < 0).astype(int)))
        ns.append(len(subset))
        valid.append(surf)

    if not valid:
        return

    x = np.arange(len(valid))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, color="darkorange", alpha=0.85, label="Model 4")
    ax.bar(x + width / 2, bases, width, color="gray", alpha=0.75, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{surf}\n(n={n:,})" for surf, n in zip(valid, ns)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.9)
    ax.set_title("Accuracy by Surface - Model 4 vs Rank Baseline (Test 2021-2024)")
    ax.legend()
    _save("accuracy_by_surface.png")


def plot_accuracy_by_confidence(y_true, probs) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_true.values})
    df["conf"] = np.abs(df["prob"] - 0.5)
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    bins = np.linspace(0, 0.5, 11)
    labels = [f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}%" for i in range(len(bins) - 1)]
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True)["correct"].agg(acc="mean", n="count").reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(grp["bin"].astype(str), grp["acc"], "o-", color="darkorange", lw=2)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.45, 1.0)
    ax1.set_title("Accuracy vs Prediction Confidence - Model 4")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.bar(grp["bin"].astype(str), grp["n"], color="steelblue", alpha=0.2)
    ax2.set_ylabel("Count")
    _save("accuracy_by_confidence.png")


def plot_prob_histogram(y_true, probs) -> None:
    df = pd.DataFrame({"prob": probs, "label": y_true.values})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[df.label == 1]["prob"], bins=30, alpha=0.6, color="darkorange", label="Wins")
    ax.hist(df[df.label == 0]["prob"], bins=30, alpha=0.6, color="steelblue", label="Losses")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution - Model 4")
    ax.legend()
    _save("prob_histogram.png")


def plot_vs_baseline_roc(y_true, probs_m4, probs_lr) -> None:
    fpr_m4, tpr_m4, _ = roc_curve(y_true, probs_m4)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, probs_lr)
    auc_m4 = roc_auc_score(y_true, probs_m4)
    auc_lr = roc_auc_score(y_true, probs_lr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_m4, tpr_m4, lw=2, color="darkorange", label=f"Model 4 (AUC={auc_m4:.4f})")
    ax.plot(fpr_lr, tpr_lr, lw=2, color="steelblue", linestyle="--", label=f"Logistic baseline (AUC={auc_lr:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison - Model 4 vs Logistic Baseline (Test 2021-2024)")
    ax.legend()
    _save("vs_baseline_roc.png")


def plot_yearly_comparison(wf: pd.DataFrame) -> None:
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m4, "o-", color="darkorange", lw=2, label="Model 4")
    ax.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=2, label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.5, label="Rank baseline")
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m4, alpha=0.10, color="darkorange", label="Model 4 gain over logistic")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Yearly Accuracy - Model 4 vs Logistic vs Rank Baseline")
    ax.legend()
    ax.set_ylim(0.55, 0.82)
    ax.grid(axis="y", alpha=0.3)
    _save("yearly_comparison.png")


def plot_feature_groups(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    totals = {}
    for group, features in GROUP_LABELS.items():
        total = 0.0
        for feature in features:
            total += imp_df.loc[imp_df["feature"].str.contains(feature, regex=False), "importance"].sum()
        totals[group] = total
    grp = pd.DataFrame({"group": list(totals.keys()), "importance": list(totals.values())})
    grp = grp.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(grp["group"], grp["importance"], color=[GROUP_COLORS_BAR[group] for group in grp["group"]])
    ax.set_xlabel("Total Feature Importance")
    ax.set_title("Feature Importance by Group - Model 4")
    _save("feature_groups.png")


def plot_elo_validation(test: pd.DataFrame) -> None:
    df = test[["elo_diff", "label"]].dropna().copy()
    bin_edges = [-600, -300, -150, -75, -25, 25, 75, 150, 300, 600]
    bin_labels = [f"{(a + b) // 2:+d}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    df["bin"] = pd.cut(df["elo_diff"], bins=bin_edges, labels=bin_labels)
    grp = df.groupby("bin", observed=True)["label"].agg(win_rate="mean", n="count").reset_index().dropna()
    theory = []
    for label in grp["bin"]:
        elo_gap = int(str(label))
        theory.append(_elo_expected(elo_gap, 0))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grp["bin"].astype(str), grp["win_rate"], "o-", color="darkorange", label="Actual")
    ax.plot(grp["bin"].astype(str), theory, "s--", color="steelblue", label="Elo theory")
    ax.set_xlabel("Elo Difference Bin")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Elo Signal Validation - Actual Win Rate vs Elo Theory (Model 4)")
    ax.legend()
    _save("elo_validation.png")


def plot_accuracy_by_round(test: pd.DataFrame, model) -> None:
    order = ["r128", "r64", "r32", "r16", "qf", "sf", "f", "rr", "br"]
    rounds = [rnd for rnd in order if rnd in test["round"].unique()]
    rounds += [
        rnd
        for rnd in test["round"].unique()
        if rnd not in rounds and rnd not in ("unknown", "nan") and pd.notna(rnd)
    ]
    valid, accs, ns = [], [], []
    for rnd in rounds:
        subset = test[test["round"] == rnd]
        if len(subset) < 50:
            continue
        probs = model.predict_proba(subset[NUMERIC_COLS_M4 + CAT_COLS])[:, 1]
        accs.append(accuracy_score(subset["label"], (probs >= 0.5).astype(int)))
        ns.append(len(subset))
        valid.append(rnd)
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(valid, accs, color="darkorange", alpha=0.85)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Accuracy by Round - Model 4 (Test 2021-2024)")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"n={n:,}", ha="center", fontsize=8)
    _save("accuracy_by_round.png")


def plot_rank_gap_accuracy(test: pd.DataFrame, probs) -> None:
    df = test[["rank_diff", "label"]].copy()
    df["prob"] = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    df["abs_gap"] = df["rank_diff"].abs()
    bins = [0, 10, 25, 50, 100, 200, 500, 2000]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500+"]
    df["gap_bin"] = pd.cut(df["abs_gap"], bins=bins, labels=labels)
    grp = (
        df.groupby("gap_bin", observed=True)
        .agg(m4=("correct", "mean"), n=("correct", "count"))
        .reset_index()
        .dropna()
    )
    rank_base = []
    for label in grp["gap_bin"]:
        subset = df[df["gap_bin"] == label]
        rank_base.append(accuracy_score(subset["label"], (subset["rank_diff"] < 0).astype(int)))
    grp["rank_base"] = rank_base

    x = np.arange(len(grp))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, grp["m4"], width, color="darkorange", alpha=0.85, label="Model 4")
    ax.bar(x + width / 2, grp["rank_base"], width, color="gray", alpha=0.75, label="Rank baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(n={n:,})" for label, n in zip(grp["gap_bin"], grp["n"])])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Accuracy by Rank Gap - Model 4 vs Rank Baseline (Test 2021-2024)")
    ax.legend()
    _save("rank_gap_accuracy.png")


def plot_injury_coverage(data: pd.DataFrame) -> None:
    coverage = data["injury_coverage"].value_counts().sort_index()
    labels_map = {0: "Neither player\nhas report", 1: "One player\nhas report", 2: "Both players\nhave reports"}
    labels = [labels_map.get(int(level), str(level)) for level in coverage.index]
    counts = coverage.values
    pct = counts / counts.sum() * 100.0
    colors = ["#d62728", "#ff7f0e", "#2ca02c"][: len(counts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.pie(pct, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Injury Report Coverage\n(% of dataset rows)")

    bars = ax2.bar(range(len(labels)), pct, color=colors, edgecolor="white", width=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("% of Rows")
    ax2.set_title("Injury Report Coverage - Count Breakdown")
    ax2.set_ylim(0, max(pct) * 1.25)
    for bar, pct_value, count in zip(bars, pct, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct_value:.1f}%\n(n={int(count):,})",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    fig.suptitle("How Much Injury Information Is Available?", fontsize=13, fontweight="bold", y=1.02)
    _save("injury_coverage.png")


def plot_injury_feature_importance(model) -> None:
    result = _extract_importances(model)
    if result is None:
        return
    names, importances = result
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(30).sort_values("importance", ascending=True)
    colors = ["crimson" if "num__inj_" in feature else "#aaaaaa" for feature in df["feature"]]

    fig, ax = plt.subplots(figsize=(11, 10))
    bars = ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Feature Importance (XGBoost Gain)")
    ax.set_title("Feature Importance - Model 4\n(crimson bars = injury features; grey = inherited Model 2 features)")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="crimson", label="Injury report features"),
            mpatches.Patch(facecolor="#aaaaaa", label="Model 2 features"),
        ],
        loc="lower right",
        fontsize=10,
    )
    for bar, feature, importance in zip(bars, df["feature"], df["importance"]):
        if "num__inj_" in feature:
            ax.text(importance + 0.0001, bar.get_y() + bar.get_height() / 2, f"  {importance:.4f}", va="center", fontsize=8.5)
    _save("injury_feature_importance.png")


def plot_m4_vs_m2_comparison(acc4, auc4, brier4, acc2, auc2, brier2, acc_lr, auc_lr, brier_lr) -> None:
    models = ["Logistic\nBaseline", "Model 2\n(no injuries)", "Model 4\n(+injuries)"]
    accs = [acc_lr, acc2, acc4]
    aucs = [auc_lr, auc2, auc4]
    briers = [brier_lr, brier2, brier4]
    colors = ["steelblue", "darkorange", "crimson"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    settings = [
        (axes[0], accs, "Accuracy\n(up is better)", "Accuracy", False),
        (axes[1], aucs, "ROC AUC\n(up is better)", "ROC AUC", False),
        (axes[2], briers, "Brier Score\n(down is better)", "Brier Score", True),
    ]
    for ax, values, title, ylabel, invert in settings:
        bars = ax.bar(models, values, color=colors, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        vmin = min(values) * 0.985
        vmax = max(values) * 1.015
        ax.set_ylim(vmin, vmax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (vmax - vmin) * 0.01, f"{value:.4f}", ha="center")
        delta = values[2] - values[1]
        good = (delta > 0 and not invert) or (delta < 0 and invert)
        sign = "+" if delta > 0 else ""
        ax.annotate(
            f"M4 vs M2: {sign}{delta:.4f}",
            xy=(2, values[2]),
            xytext=(1.5, vmax * 0.997),
            fontsize=9,
            color="green" if good else "red",
            fontweight="bold",
            ha="center",
        )

    fig.suptitle("Model 4 vs Model 2 vs Logistic Baseline\n(Test 2021-2024 - same split)", fontsize=13, fontweight="bold")
    _save("m4_vs_m2_comparison.png")


def plot_m4_vs_m2_yearly(wf: pd.DataFrame, acc_m2_holdout: float) -> None:
    if wf.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wf.test_year, wf.acc_m4, "o-", color="crimson", lw=2, label="Model 4 - XGBoost + injuries")
    ax.plot(wf.test_year, wf.acc_lr, "s--", color="steelblue", lw=1.5, label="Logistic baseline")
    ax.plot(wf.test_year, wf.rank_baseline, "^:", color="gray", lw=1.2, label="Rank baseline")
    ax.axhline(
        acc_m2_holdout,
        color="darkorange",
        linestyle="-.",
        lw=1.5,
        alpha=0.8,
        label=f"Model 2 holdout accuracy ({acc_m2_holdout:.4f})",
    )
    ax.fill_between(wf.test_year, wf.acc_lr, wf.acc_m4, alpha=0.12, color="crimson", label="M4 injury gain over logistic")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-Forward Accuracy - Model 4 vs Model 2 reference\n(orange line = Model 2 final holdout accuracy)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.55, 0.82)
    ax.grid(axis="y", alpha=0.3)
    _save("m4_vs_m2_yearly.png")


def _injury_bucket(days_since_last: float, recent_flag: float, has_report: float) -> str:
    if has_report < 0.5:
        return "No report"
    if recent_flag < 0.5:
        return "No injury\nlast 365d"
    if days_since_last <= 30:
        return "0-30 days"
    if days_since_last <= 90:
        return "31-90 days"
    if days_since_last <= 180:
        return "91-180 days"
    return "181-365 days"


def plot_injury_recency_story(test: pd.DataFrame, probs: np.ndarray) -> None:
    df = test.copy()
    df["prob"] = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    df["bucket"] = [
        _injury_bucket(days, recent, report)
        for days, recent, report in zip(
            df["p1_inj_days_since_last"],
            df["p1_inj_recent_flag"],
            df["p1_has_injury_report"],
        )
    ]
    order = ["No injury\nlast 365d", "181-365 days", "91-180 days", "31-90 days", "0-30 days"]
    grp = (
        df[df["bucket"] != "No report"]
        .groupby("bucket", observed=True)
        .agg(win_rate=("label", "mean"), accuracy=("correct", "mean"), n=("label", "count"))
        .reindex(order)
        .dropna()
        .reset_index()
    )
    if grp.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = ax1.bar(grp["bucket"], grp["win_rate"], color="crimson", alpha=0.85)
    ax1.axhline(0.5, color="gray", linestyle=":")
    ax1.set_ylabel("Actual P1 Win Rate")
    ax1.set_ylim(0.25, 0.75)
    ax1.set_title("How does recent injury timing affect match results?")
    ax1.tick_params(axis="x", rotation=20)
    for bar, n in zip(bars1, grp["n"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n:,}", ha="center", fontsize=8)

    bars2 = ax2.bar(grp["bucket"], grp["accuracy"], color="darkorange", alpha=0.85)
    ax2.axhline(df["correct"].mean(), color="gray", linestyle=":", label="Overall accuracy")
    ax2.set_ylabel("Model 4 Accuracy")
    ax2.set_ylim(0.45, 0.85)
    ax2.set_title("Does Model 4 help more on recently injured players?")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend()
    for bar, value in zip(bars2, grp["accuracy"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontsize=8)

    _save("injury_recency_story.png")


def plot_injury_type_impact(test: pd.DataFrame, probs: np.ndarray) -> None:
    df = test.copy()
    df["prob"] = probs
    categories = {
        "Any recent injury": df["p1_inj_recent_flag"] == 1,
        "Recent joint injury": df["p1_inj_joint_recent"] == 1,
        "Recent chronic issue": df["p1_inj_chronic_recent"] == 1,
        "Recent injury retirement": df["p1_inj_retired_recent"] == 1,
        "Returned in last 90d": df["p1_inj_recent_return"] == 1,
    }
    rows = []
    for label, mask in categories.items():
        subset = df[mask]
        if len(subset) < 20:
            continue
        rows.append(
            {
                "category": label,
                "win_rate": subset["label"].mean(),
                "pred_prob": subset["prob"].mean(),
                "n": len(subset),
            }
        )
    story = pd.DataFrame(rows)
    if story.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(story))
    width = 0.36
    ax.bar(x - width / 2, story["win_rate"], width, color="crimson", alpha=0.85, label="Actual win rate")
    ax.bar(x + width / 2, story["pred_prob"], width, color="darkorange", alpha=0.85, label="Mean predicted win prob")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cat}\n(n={n:,})" for cat, n in zip(story["category"], story["n"])], rotation=15)
    ax.set_ylabel("Rate / Probability")
    ax.set_ylim(0.20, 0.80)
    ax.set_title("Which injury situations matter most in tennis?")
    ax.legend()
    _save("injury_type_impact.png")


def plot_injury_coverage_accuracy(test: pd.DataFrame, probs: np.ndarray) -> None:
    df = test.copy()
    df["prob"] = probs
    df["correct"] = ((df["prob"] >= 0.5).astype(int) == df["label"]).astype(int)
    labels_map = {0: "Neither\nplayer", 1: "One\nplayer", 2: "Both\nplayers"}
    accs, labels = [], []
    for level in [0, 1, 2]:
        subset = df[df["injury_coverage"] == level]
        if len(subset) < 20:
            continue
        accs.append(subset["correct"].mean())
        labels.append(f"{labels_map[level]}\n(n={len(subset):,})")
    if not accs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(accs)), accs, color=["#d62728", "#ff7f0e", "#2ca02c"][: len(accs)], edgecolor="white", width=0.5)
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 0.85)
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_title("Model 4 Accuracy by Injury Report Availability")
    for bar, value in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{value:.3f}", ha="center", fontsize=12, fontweight="bold")
    _save("injury_coverage_accuracy.png")


def main() -> None:
    print("\n" + "=" * 68)
    print("  Model 4 - XGBoost + Elo + Rolling Form + Injury Reports")
    print("=" * 68)

    print("\nLoading data ...")
    processed = load_processed(PROC_DIR)
    matches = load_initial_matches(INIT_DIR)
    surf_df = load_surface_results(INIT_DIR)
    injury_lookup = load_injury_data(INJURY_DIR)

    print("\nBuilding static lookups ...")
    player_attrs = build_player_attrs(matches)
    surf_lookup = build_surface_lookup(surf_df)

    print("\nComputing dynamic features (Elo, form, H2H) ...")
    processed = compute_dynamic_features(processed)

    print("\nBuilding symmetric dataset ...")
    data = build_dataset(processed, player_attrs, surf_lookup, injury_lookup)
    data_comp = data[(data["year"] >= 2014) & (data["year"] <= 2024)].copy()

    print("\nRunning walk-forward validation (2016-2024) ...")
    wf = walk_forward(data_comp)

    print("\nTraining final models (train 2014-2020 / test 2021-2024) ...")
    train = data_comp[data_comp["year"] <= 2020]
    test = data_comp[data_comp["year"] >= 2021]

    model_m4 = make_m4_pipeline()
    acc4, auc4, brier4, probs4 = fit_eval(
        model_m4,
        train[NUMERIC_COLS_M4 + CAT_COLS],
        train["label"],
        test[NUMERIC_COLS_M4 + CAT_COLS],
        test["label"],
    )

    print("  Training M2-equivalent (no injuries) for comparison ...")
    model_m2 = make_m2_pipeline()
    acc2, auc2, brier2, probs2 = fit_eval(
        model_m2,
        train[NUMERIC_COLS_M2 + CAT_COLS],
        train["label"],
        test[NUMERIC_COLS_M2 + CAT_COLS],
        test["label"],
    )

    lr_model = make_baseline_pipeline()
    acc_lr, auc_lr, brier_lr, probs_lr = fit_eval(
        lr_model,
        train[BASELINE_NUM + BASELINE_CAT],
        train["label"],
        test[BASELINE_NUM + BASELINE_CAT],
        test["label"],
    )

    rank_base = accuracy_score(test["label"], (test["rank_diff"] < 0).astype(int))

    print(f"\n{'=' * 74}")
    print("  Results - Model 4 vs Model 2 vs Logistic Baseline vs Rank Baseline")
    print(f"{'=' * 74}")
    print(f"{'Metric':<28} {'Model 4':>12}  {'Model 2':>10}  {'Logistic LR':>12}  {'Rank base':>10}")
    print(f"{'-' * 74}")
    print(f"{'Accuracy':<28} {acc4:>12.4f}  {acc2:>10.4f}  {acc_lr:>12.4f}  {rank_base:>10.4f}")
    print(f"{'ROC AUC':<28} {auc4:>12.4f}  {auc2:>10.4f}  {auc_lr:>12.4f}  {'-':>10}")
    print(f"{'Brier Score':<28} {brier4:>12.4f}  {brier2:>10.4f}  {brier_lr:>12.4f}  {'-':>10}  (down is better)")
    print(f"{'-' * 74}")
    print(f"{'Delta Accuracy  M4 vs M2':<28} {acc4 - acc2:>+12.4f}")
    print(f"{'Delta AUC       M4 vs M2':<28} {auc4 - auc2:>+12.4f}")
    print(f"{'Delta Brier     M4 vs M2':<28} {brier4 - brier2:>+12.4f}  (negative = better)")
    print(f"{'-' * 74}")
    print(f"{'Delta Accuracy  M4 vs LR':<28} {acc4 - acc_lr:>+12.4f}")
    print(f"{'Delta AUC       M4 vs LR':<28} {auc4 - auc_lr:>+12.4f}")
    print(f"{'=' * 74}\n")

    joblib.dump(model_m4, os.path.join(ROOT, "model_4.joblib"))
    print(f"Saved model -> {os.path.join(ROOT, 'model_4.joblib')}")

    print("\nGenerating plots ...")
    plot_roc_curve(test["label"], probs4, acc4, auc4)
    plot_calibration(test["label"], probs4)
    plot_confusion_matrix_chart(test["label"], probs4)
    plot_feature_importance(model_m4)
    plot_walk_forward(wf)
    plot_accuracy_by_surface(test, model_m4)
    plot_accuracy_by_confidence(test["label"], probs4)
    plot_prob_histogram(test["label"], probs4)
    plot_vs_baseline_roc(test["label"], probs4, probs_lr)
    plot_yearly_comparison(wf)
    plot_feature_groups(model_m4)
    plot_elo_validation(test)
    plot_accuracy_by_round(test, model_m4)
    plot_rank_gap_accuracy(test, probs4)

    plot_injury_coverage(data_comp)
    plot_injury_feature_importance(model_m4)
    plot_m4_vs_m2_comparison(acc4, auc4, brier4, acc2, auc2, brier2, acc_lr, auc_lr, brier_lr)
    plot_m4_vs_m2_yearly(wf, acc2)
    plot_injury_recency_story(test, probs4)
    plot_injury_type_impact(test, probs4)
    plot_injury_coverage_accuracy(test, probs4)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
