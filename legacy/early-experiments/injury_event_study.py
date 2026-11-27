#!/usr/bin/env python3
"""
injury_event_study.py

Event-study analysis of player performance before and after documented injuries.

Rather than using injury as a prediction feature, this script measures how
much injuries actually impact player performance by comparing:
  - Win rate in the N matches BEFORE injury date
  - Win rate in the N matches AFTER estimated return date

This is a causal / descriptive analysis, not a model training script.

Produces 5 charts in plots_injury_impact/:
  1. event_study.png           — win rate at each match index -10 to +10 relative to return
  2. pre_post_by_type.png      — aggregate pre vs post win rate by injury type
  3. timeline_30d_buckets.png  — win rate in 30-day buckets around return date
  4. recovery_arc_by_type.png  — event study split by joint / chronic / acute
  5. winrate_change_dist.png   — distribution of (post − pre) win rate per injury event

Usage:
    python injury_event_study.py
    python injury_event_study.py --injury-dir injury_reports --window 10

Requirements: pandas, numpy, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

OUT_DIR        = "plots_injury_impact"
MATCH_WINDOW   = 10    # matches before/after to include in event study
PRE_DAYS       = 180   # max days before injury to look for pre-injury matches
POST_DAYS      = 365   # max days after return to look for post-return matches
MIN_PRE        = 3     # minimum pre-injury matches needed to include an event
MIN_POST       = 3     # minimum post-return matches needed to include an event


# ── Injury helpers (same as advanced_model_injury.py) ─────────────────────────

def _parse_injury_date(s):
    if not s or str(s).strip().lower() == "unknown":
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def _parse_recovery_days(s):
    if not s or str(s).strip().lower() in ("unknown", ""):
        return None
    s = str(s).strip().lower().replace("about ", "").strip()

    def _to_days(val, unit):
        if "month" in unit:
            return val * 30.0
        if "week" in unit:
            return val * 7.0
        return val

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", s)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        return _to_days((lo + hi) / 2.0, unit)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", s)
    if m:
        return _to_days(float(m.group(1)), m.group(2))

    if "same day" in s:
        return 2.0
    if "day" in s and "week" in s:
        return 7.0
    if "day" in s:
        return 2.0
    if "week" in s:
        return 14.0
    return None


def _norm_name(name):
    return str(name).strip().lower().replace(".", "").replace("-", " ").replace("  ", " ").strip()


def _csv_key(full_name):
    """'Grigor Dimitrov' → 'dimitrov g'  (matches ATP CSV abbreviation style)"""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    initial  = parts[0][0].lower()
    lastname = " ".join(parts[1:]).lower().replace("-", " ")
    return f"{lastname} {initial}"


def load_injury_events(injury_dir: str) -> list[dict]:
    """
    Load all injury events with parseable recovery times.
    Returns list of dicts with keys:
      player_full, player_csv, injury_date, return_date, recovery_days,
      joint, chronic, acute, retired
    """
    events = []
    files  = sorted(_glob.glob(os.path.join(injury_dir, "*.json")))

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for player_name, pdata in raw.get("players", {}).items():
            csv_key = _csv_key(player_name)
            for inj in pdata.get("injuries", []):
                inj_date = _parse_injury_date(inj.get("injury_date"))
                if inj_date is None:
                    continue
                rec_days = _parse_recovery_days(inj.get("estimated_recovery_time"))
                if rec_days is None:
                    continue
                return_date = inj_date + pd.Timedelta(days=rec_days)
                retired_raw = str(inj.get("retired_during_match_due_to_injury", "no"))
                events.append({
                    "player_full": player_name,
                    "player_csv":  csv_key,
                    "injury_date": inj_date,
                    "return_date": return_date,
                    "recovery_days": rec_days,
                    "joint":   bool(inj.get("joint_injury", False)),
                    "chronic": inj.get("acute_or_chronic") == "chronic",
                    "acute":   inj.get("acute_or_chronic") == "acute",
                    "retired": retired_raw.strip().lower() in ("yes", "true", "1"),
                })

    print(f"  Loaded {len(events)} injury events with known recovery times "
          f"from {len(files)} files")
    return events


# ── Match history builder ─────────────────────────────────────────────────────

def build_player_histories(csv_path: str) -> dict[str, list[tuple]]:
    """
    Returns dict: normalized_player_name → sorted list of (date, won: bool)
    Uses the same _norm_name normalization as the injury DB lookup.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player_1", "Player_2", "Winner"])

    histories: dict[str, list] = defaultdict(list)
    for row in df.itertuples(index=False):
        p1   = str(row.Player_1)
        p2   = str(row.Player_2)
        win  = str(row.Winner)
        date = row.Date
        k1   = _norm_name(p1)
        k2   = _norm_name(p2)
        histories[k1].append((date, win == p1))
        histories[k2].append((date, win == p2))

    for k in histories:
        histories[k].sort(key=lambda x: x[0])

    print(f"  {len(histories):,} unique players with match histories")
    return dict(histories)


# ── Event-study extraction ────────────────────────────────────────────────────

def extract_events(
    injury_events: list[dict],
    histories: dict[str, list],
    match_window: int,
    pre_days: int,
    post_days: int,
    min_pre: int,
    min_post: int,
) -> pd.DataFrame:
    """
    For each injury event, find pre-injury and post-return matches and
    assign a relative position index.

    Returns a DataFrame with columns:
      event_id, player, position (negative = pre, positive = post),
      won, days_from_return, joint, chronic, acute, retired,
      recovery_days, pre_winrate, post_winrate
    """
    rows    = []
    n_used  = 0
    n_skip  = 0

    for eid, ev in enumerate(injury_events):
        csv_key  = ev["player_csv"]
        if csv_key is None:
            n_skip += 1
            continue

        history = histories.get(csv_key, [])
        if not history:
            # Also try looking up by first initial differently
            n_skip += 1
            continue

        inj_date    = ev["injury_date"]
        return_date = ev["return_date"]
        pre_cutoff  = inj_date  - pd.Timedelta(days=pre_days)
        post_cutoff = return_date + pd.Timedelta(days=post_days)

        # Matches in pre-injury window (before injury, within pre_days)
        pre = [(d, w) for d, w in history
               if pre_cutoff <= d < inj_date]
        # Matches in post-return window (after return, within post_days)
        post = [(d, w) for d, w in history
                if return_date <= d < post_cutoff]

        if len(pre) < min_pre or len(post) < min_post:
            n_skip += 1
            continue

        # Sort pre most-recent-first, take up to match_window
        pre.sort(key=lambda x: x[0], reverse=True)
        pre = pre[:match_window]

        # Sort post chronological, take up to match_window
        post.sort(key=lambda x: x[0])
        post = post[:match_window]

        pre_winrate  = float(np.mean([w for _, w in pre]))
        post_winrate = float(np.mean([w for _, w in post]))

        meta = dict(
            event_id      = eid,
            player        = ev["player_full"],
            joint         = ev["joint"],
            chronic       = ev["chronic"],
            acute         = ev["acute"],
            retired       = ev["retired"],
            recovery_days = ev["recovery_days"],
            pre_winrate   = pre_winrate,
            post_winrate  = post_winrate,
        )

        for i, (d, w) in enumerate(pre):
            rows.append({**meta,
                "position":        -(i + 1),
                "won":             int(w),
                "days_from_return": (d - return_date).days,
            })
        for i, (d, w) in enumerate(post):
            rows.append({**meta,
                "position":        i + 1,
                "won":             int(w),
                "days_from_return": (d - return_date).days,
            })

        n_used += 1

    print(f"  Events used: {n_used}  |  Skipped (no match history or too few matches): {n_skip}")
    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _ci95(series):
    """Return (mean, lower_95, upper_95) using normal approximation."""
    n   = len(series)
    mu  = series.mean()
    if n < 2:
        return mu, mu, mu
    se  = series.std(ddof=1) / np.sqrt(n)
    z   = 1.96
    return mu, mu - z * se, mu + z * se


# ── Chart 1 — Event study ─────────────────────────────────────────────────────

def plot_event_study(df: pd.DataFrame, out_dir: str):
    """
    Win rate at each match position relative to return from injury.
    Position -1 = match immediately before injury, +1 = first match after return.
    """
    positions = sorted(df["position"].unique())
    means, lows, highs = [], [], []

    for pos in positions:
        sub  = df[df["position"] == pos]["won"]
        mu, lo, hi = _ci95(sub)
        means.append(mu)
        lows.append(lo)
        highs.append(hi)

    means  = np.array(means)
    lows   = np.array(lows)
    highs  = np.array(highs)
    x      = np.array(positions)

    # Baseline: overall pre-injury win rate (positions < 0)
    pre_mask      = df["position"] < 0
    baseline_rate = df[pre_mask]["won"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Shade pre vs post regions
    ax.axvspan(x.min() - 0.5, -0.5, alpha=0.06, color="steelblue", label="_nolegend_")
    ax.axvspan(0.5, x.max() + 0.5, alpha=0.06, color="tomato",    label="_nolegend_")

    # Confidence band
    ax.fill_between(x, lows, highs, alpha=0.2, color="darkorange")

    # Win rate line
    pre_x  = x[x < 0];  pre_y  = means[x < 0]
    post_x = x[x > 0];  post_y = means[x > 0]
    ax.plot(pre_x,  pre_y,  "o-", color="steelblue",  lw=2, label="Pre-injury matches")
    ax.plot(post_x, post_y, "s-", color="darkorange",  lw=2, label="Post-return matches")

    # Baseline reference
    ax.axhline(baseline_rate, color="steelblue", linestyle="--", lw=1.2, alpha=0.7,
               label=f"Pre-injury avg ({baseline_rate:.3f})")

    # 50% chance reference
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)

    # Vertical line at injury / return
    ax.axvline(0, color="black", linestyle="-", lw=1.5, alpha=0.5)
    ax.text(0.02, 0.97, "← Pre-injury   |   Post-return →",
            transform=ax.transAxes, va="top", fontsize=10, color="gray")

    # Sample sizes per position
    n_events = df.groupby("position")["event_id"].nunique()
    for pos in positions:
        n = n_events.get(pos, 0)
        ax.annotate(f"n={n}", xy=(pos, lows[positions.index(pos)] - 0.015),
                    ha="center", fontsize=7, color="gray")

    ax.set_xlabel("Match index relative to injury return\n(−1 = last match before injury, +1 = first match after return)")
    ax.set_ylabel("Win rate")
    ax.set_title("Player Performance Before and After Injury Return\n(event study across all injury events)")
    ax.set_xticks(x)
    ax.set_ylim(0.2, 0.9)
    ax.legend(loc="upper right")
    _savefig(os.path.join(out_dir, "event_study.png"))


# ── Chart 2 — Pre vs Post by injury type ─────────────────────────────────────

def plot_pre_post_by_type(df: pd.DataFrame, out_dir: str):
    """
    Aggregate pre vs post win rate grouped by injury type.
    Each group shows: mean pre win rate vs mean post win rate, with 95% CI.
    """
    # Build per-event summary (one row per event)
    event_summary = (
        df.groupby("event_id")
          .agg(
              pre_wr   = ("pre_winrate",  "first"),
              post_wr  = ("post_winrate", "first"),
              joint    = ("joint",        "first"),
              chronic  = ("chronic",      "first"),
              acute    = ("acute",        "first"),
              retired  = ("retired",      "first"),
          )
          .reset_index()
    )

    categories = {
        "All injuries":    event_summary,
        "Joint injuries":  event_summary[event_summary["joint"]],
        "Chronic":         event_summary[event_summary["chronic"]],
        "Acute":           event_summary[event_summary["acute"]],
        "Retired in match":event_summary[event_summary["retired"]],
    }
    # Only include categories with enough events
    categories = {k: v for k, v in categories.items() if len(v) >= 3}

    labels = list(categories.keys())
    pre_means, pre_los, pre_his = [], [], []
    pst_means, pst_los, pst_his = [], [], []
    ns = []

    for lbl in labels:
        sub = categories[lbl]
        ns.append(len(sub))
        mu, lo, hi = _ci95(sub["pre_wr"]);  pre_means.append(mu); pre_los.append(lo); pre_his.append(hi)
        mu, lo, hi = _ci95(sub["post_wr"]); pst_means.append(mu); pst_los.append(lo); pst_his.append(hi)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_pre  = ax.bar(x - w/2, pre_means, w, color="steelblue",  alpha=0.85, label="Pre-injury win rate")
    bars_post = ax.bar(x + w/2, pst_means, w, color="darkorange", alpha=0.85, label="Post-return win rate")

    # Error bars
    pre_err = [np.array(pre_means) - np.array(pre_los),
               np.array(pre_his)   - np.array(pre_means)]
    pst_err = [np.array(pst_means) - np.array(pst_los),
               np.array(pst_his)   - np.array(pst_means)]
    ax.errorbar(x - w/2, pre_means, yerr=pre_err, fmt="none", color="black", capsize=4)
    ax.errorbar(x + w/2, pst_means, yerr=pst_err, fmt="none", color="black", capsize=4)

    # Delta annotations
    for i, (pre, pst) in enumerate(zip(pre_means, pst_means)):
        delta = pst - pre
        color = "red" if delta < 0 else "green"
        ax.annotate(f"Δ {delta:+.3f}", xy=(x[i], max(pre, pst) + 0.03),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=10)
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pre-injury vs Post-return Win Rate by Injury Type\n(error bars = 95% CI)")
    ax.legend()
    _savefig(os.path.join(out_dir, "pre_post_by_type.png"))


# ── Chart 3 — 30-day bucket timeline ─────────────────────────────────────────

def plot_timeline_buckets(df: pd.DataFrame, out_dir: str):
    """
    Win rate in 30-day buckets centred on return date.
    Shows the performance arc from -180 days to +365 days.
    Captures the gradual dip and recovery curve.
    """
    bucket_size = 30
    min_days    = -180
    max_days    = 365
    edges       = range(min_days, max_days + bucket_size, bucket_size)
    labels_x    = []
    means_y, lows_y, highs_y, ns_y = [], [], [], []

    for lo in range(min_days, max_days, bucket_size):
        hi  = lo + bucket_size
        sub = df[(df["days_from_return"] >= lo) & (df["days_from_return"] < hi)]["won"]
        if len(sub) < 3:
            continue
        mu, low, high = _ci95(sub)
        centre = (lo + hi) / 2
        labels_x.append(centre)
        means_y.append(mu)
        lows_y.append(low)
        highs_y.append(high)
        ns_y.append(len(sub))

    labels_x = np.array(labels_x)
    means_y  = np.array(means_y)
    lows_y   = np.array(lows_y)
    highs_y  = np.array(highs_y)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Shade pre vs post
    ax.axvspan(labels_x.min() - 15, 0, alpha=0.06, color="steelblue")
    ax.axvspan(0, labels_x.max() + 15, alpha=0.06, color="tomato")

    ax.fill_between(labels_x, lows_y, highs_y, alpha=0.2, color="darkorange")
    ax.plot(labels_x, means_y, "o-", color="darkorange", lw=2)

    # Pre-injury baseline
    pre_bl = df[df["days_from_return"] < 0]["won"].mean()
    ax.axhline(pre_bl, color="steelblue", linestyle="--", lw=1.5,
               label=f"Pre-injury avg ({pre_bl:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.axvline(0, color="black", lw=1.5, alpha=0.5, label="Estimated return date")

    # Annotate sample sizes
    for x_val, n in zip(labels_x, ns_y):
        ax.annotate(f"{n}", xy=(x_val, lows_y[list(labels_x).index(x_val)] - 0.02),
                    ha="center", fontsize=7, color="gray")

    ax.set_xlabel("Days from estimated return date\n(negative = still before return / pre-injury period)")
    ax.set_ylabel("Win rate (30-day rolling bucket)")
    ax.set_title("Player Win Rate Timeline Relative to Injury Return\n(each dot = 30-day window, n = matches in bucket)")
    ax.legend()
    ax.set_ylim(0.2, 0.9)
    _savefig(os.path.join(out_dir, "timeline_30d_buckets.png"))


# ── Chart 4 — Recovery arc by injury type ────────────────────────────────────

def plot_recovery_arc_by_type(df: pd.DataFrame, out_dir: str):
    """
    Event study (match index) split by injury type.
    Shows if joint injuries cause a longer/deeper performance dip than non-joint.
    """
    groups = {
        "Joint injury":     df[df["joint"]   == True],
        "Non-joint injury": df[df["joint"]   == False],
        "Chronic":          df[df["chronic"] == True],
        "Acute":            df[df["chronic"] == False],
    }
    groups = {k: v for k, v in groups.items() if len(v[v["position"] > 0]) >= 10}

    positions = sorted(df["position"].unique())
    colors    = ["tomato", "steelblue", "darkorange", "mediumseagreen"]

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 6), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, (lbl, sub), color in zip(axes, groups.items(), colors):
        means, lows, highs = [], [], []
        for pos in positions:
            s = sub[sub["position"] == pos]["won"]
            if len(s) == 0:
                means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            mu, lo, hi = _ci95(s)
            means.append(mu); lows.append(lo); highs.append(hi)

        x = np.array(positions)
        means = np.array(means); lows = np.array(lows); highs = np.array(highs)
        valid = ~np.isnan(means)

        pre_bl = sub[sub["position"] < 0]["won"].mean()

        ax.axvspan(x.min() - 0.5, -0.5, alpha=0.06, color="steelblue")
        ax.axvspan(0.5, x.max() + 0.5, alpha=0.06, color="tomato")
        ax.fill_between(x[valid], lows[valid], highs[valid], alpha=0.2, color=color)
        ax.plot(x[valid], means[valid], "o-", color=color, lw=2)
        ax.axhline(pre_bl, color="steelblue", linestyle="--", lw=1.2, alpha=0.7)
        ax.axhline(0.5,    color="gray",      linestyle=":",  lw=1)
        ax.axvline(0,      color="black",     lw=1.5, alpha=0.4)

        n_events = sub["event_id"].nunique()
        ax.set_title(f"{lbl}\n(n={n_events} events)")
        ax.set_xlabel("Match index (relative to return)")
        ax.set_xticks(x)
        ax.set_ylim(0.2, 0.9)
        if ax is axes[0]:
            ax.set_ylabel("Win rate")

    fig.suptitle("Recovery Arc by Injury Type", fontsize=13, y=1.02)
    _savefig(os.path.join(out_dir, "recovery_arc_by_type.png"))


# ── Chart 5 — Distribution of win rate change ─────────────────────────────────

def plot_winrate_change_dist(df: pd.DataFrame, out_dir: str):
    """
    Histogram of (post_winrate − pre_winrate) per injury event.
    Shows whether the performance drop is consistent or driven by a few outliers.
    Also does a one-sample t-test: is the mean change significantly different from 0?
    """
    # One row per event
    events = (
        df.groupby("event_id")
          .agg(delta    = ("post_winrate", "first"),
               pre_wr   = ("pre_winrate",  "first"),
               post_wr  = ("post_winrate", "first"),
               player   = ("player",       "first"),
               joint    = ("joint",        "first"),
               chronic  = ("chronic",      "first"))
          .assign(delta=lambda x: x["post_wr"] - x["pre_wr"])
          .reset_index()
    )

    deltas = events["delta"].dropna()
    mean_d = deltas.mean()
    t_stat, p_val = stats.ttest_1samp(deltas, 0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Separate positive and negative changes
    neg = deltas[deltas < 0]
    pos = deltas[deltas >= 0]
    bins = np.linspace(-1, 1, 21)

    ax.hist(neg, bins=bins, color="tomato",      alpha=0.8, label=f"Performance dropped ({len(neg)} events)")
    ax.hist(pos, bins=bins, color="mediumseagreen", alpha=0.8, label=f"Performance held/improved ({len(pos)} events)")

    ax.axvline(0,      color="black",     lw=1.5, linestyle="-")
    ax.axvline(mean_d, color="darkred",   lw=2,   linestyle="--",
               label=f"Mean Δ = {mean_d:+.3f}")

    # Significance annotation
    sig_str = (f"t={t_stat:.2f},  p={p_val:.3f}"
               + (" *" if p_val < 0.05 else "  (not significant)"))
    ax.text(0.97, 0.95, sig_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    ax.set_xlabel("Post-return win rate  −  Pre-injury win rate")
    ax.set_ylabel("Number of injury events")
    ax.set_title("Distribution of Win Rate Change After Injury Return\n"
                 "(negative = performance dropped post-return)")
    ax.legend()
    _savefig(os.path.join(out_dir, "winrate_change_dist.png"))

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"  Win rate change summary ({len(deltas)} injury events)")
    print(f"{'─'*55}")
    print(f"  Mean Δ (post − pre):  {mean_d:+.4f}")
    print(f"  Median Δ:             {deltas.median():+.4f}")
    print(f"  Std dev:              {deltas.std():.4f}")
    print(f"  % events with drop:   {(deltas < 0).mean()*100:.1f}%")
    print(f"  t-statistic:          {t_stat:.3f}")
    print(f"  p-value:              {p_val:.4f}  {'← significant at 5%' if p_val < 0.05 else ''}")
    print(f"{'─'*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Injury event study analysis.")
    p.add_argument("--data-path",   default="atp_tennis.csv")
    p.add_argument("--injury-dir",  default="injury_reports")
    p.add_argument("--window",      type=int, default=MATCH_WINDOW,
                   help="Matches before/after to include in event study")
    p.add_argument("--pre-days",    type=int, default=PRE_DAYS)
    p.add_argument("--post-days",   type=int, default=POST_DAYS)
    p.add_argument("--plots-dir",   default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\nLoading match data from '{args.data_path}' ...")
    histories = build_player_histories(args.data_path)

    print(f"\nLoading injury events from '{args.injury_dir}/' ...")
    injury_events = load_injury_events(args.injury_dir)

    print(f"\nBuilding event study (±{args.window} matches around return date) ...")
    df = extract_events(
        injury_events, histories,
        match_window=args.window,
        pre_days=args.pre_days,
        post_days=args.post_days,
        min_pre=MIN_PRE,
        min_post=MIN_POST,
    )

    if df.empty:
        print("  No events extracted — check name matching or data paths.")
        return

    n_events = df["event_id"].nunique()
    print(f"  {n_events} injury events with sufficient pre/post match data")
    print(f"  {len(df)} total match-position observations")

    print(f"\nGenerating charts in '{args.plots_dir}/' ...")
    plot_event_study(df, args.plots_dir)
    plot_pre_post_by_type(df, args.plots_dir)
    plot_timeline_buckets(df, args.plots_dir)
    plot_recovery_arc_by_type(df, args.plots_dir)
    plot_winrate_change_dist(df, args.plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
