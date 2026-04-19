#!/usr/bin/env python3
"""
Linear regression model to predict match scores and winner.

Approach:
- Use only pre-match features (rank, rank points, age, height, surface, etc.)
- Parse the match score into total games for winner/loser
- Create two rows per match (winner as player1, loser as player1) to keep it symmetric
- Train a linear regression model to predict total games for player1 and player2
- Derive predicted winner by comparing predicted games
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


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


def load_matches(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    dfs = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def parse_score(score: str):
    if not isinstance(score, str):
        return None
    s = score.strip()
    if not s:
        return None
    if any(ch.isalpha() for ch in s):
        return None

    pairs = re.findall(r"(\d+)-(\d+)", s)
    if not pairs:
        return None

    winner_games = 0
    loser_games = 0
    for a, b in pairs:
        winner_games += int(a)
        loser_games += int(b)

    return winner_games, loser_games


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "tourney_date",
        "surface",
        "tourney_level",
        "draw_size",
        "best_of",
        "round",
        "score",
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

    df = df.dropna(
        subset=[
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
            "score",
        ]
    )

    # Parse score into total games for winner/loser
    parsed = df["score"].apply(parse_score)
    df = df[parsed.notna()].copy()
    scores = parsed[parsed.notna()].tolist()
    df["winner_games"] = [x[0] for x in scores]
    df["loser_games"] = [x[1] for x in scores]

    # Winner as player1
    w = pd.DataFrame(
        {
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
            "p1_games": df["winner_games"],
            "p2_games": df["loser_games"],
        }
    )

    # Loser as player1 (mirror row)
    l = pd.DataFrame(
        {
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
            "p1_games": df["loser_games"],
            "p2_games": df["winner_games"],
        }
    )

    data = pd.concat([w, l], ignore_index=True)

    # Feature engineering
    data["rank_diff"] = data["p1_rank"] - data["p2_rank"]
    data["rank_points_diff"] = data["p1_rank_points"] - data["p2_rank_points"]
    data["seed_diff"] = data["p1_seed"] - data["p2_seed"]
    data["age_diff"] = data["p1_age"] - data["p2_age"]
    data["ht_diff"] = data["p1_ht"] - data["p2_ht"]

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
    y_train = train[["p1_games", "p2_games"]]
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]
    y_test = test[["p1_games", "p2_games"]]
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

    reg = LinearRegression()
    return Pipeline(steps=[("preprocess", preprocessor), ("reg", reg)])


def evaluate(model: Pipeline, X_test, y_test):
    preds = model.predict(X_test)
    pred_p1 = preds[:, 0]
    pred_p2 = preds[:, 1]

    true_p1 = y_test["p1_games"].values
    true_p2 = y_test["p2_games"].values

    mae_p1 = mean_absolute_error(true_p1, pred_p1)
    mae_p2 = mean_absolute_error(true_p2, pred_p2)
    mae_avg = mean_absolute_error(y_test, preds)

    pred_winner = pred_p1 >= pred_p2
    true_winner = true_p1 >= true_p2
    win_acc = accuracy_score(true_winner, pred_winner)

    return mae_p1, mae_p2, mae_avg, win_acc


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a linear regression model to predict scores and winner."
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
    parser.add_argument("--save-model", default="model_linear_score.joblib")
    args = parser.parse_args()

    return Config(
        data_glob=args.data_glob,
        train_year_start=args.train_year_start,
        train_year_end=args.train_year_end,
        test_year_start=args.test_year_start,
        test_year_end=args.test_year_end,
        save_model=args.save_model,
    )


def main():
    cfg = parse_args()

    df = load_matches(cfg.data_glob)
    data = build_dataset(df)

    X_train, X_test, y_train, y_test = make_splits(data, cfg)

    model = build_model()
    model.fit(X_train, y_train)

    mae_p1, mae_p2, mae_avg, win_acc = evaluate(model, X_test, y_test)
    print(f"MAE p1_games: {mae_p1:.3f}")
    print(f"MAE p2_games: {mae_p2:.3f}")
    print(f"MAE average:  {mae_avg:.3f}")
    print(f"Winner accuracy (from predicted games): {win_acc:.4f}")

    if cfg.save_model:
        joblib.dump(model, cfg.save_model)
        print(f"Saved model to: {cfg.save_model}")


if __name__ == "__main__":
    main()
