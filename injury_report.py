from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = BASE_DIR / "injury_reports_top10.json"
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_RANKINGS_FILE = BASE_DIR / "tennis_atp" / "atp_rankings_current.csv"
DEFAULT_PLAYERS_FILE = BASE_DIR / "tennis_atp" / "atp_players.csv"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TOP_N = 10
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_OUTPUT_TOKENS = 2500
DEFAULT_MATCH_GLOB = "atp_matches_*.csv"
MAX_RETRIES = 3

PRIMARY_SOURCE_DOMAINS = [
    "apnews.com",
    "atptour.com",
    "ausopen.com",
    "bbc.com",
    "espn.com",
    "olympics.com",
    "reuters.com",
    "rolandgarros.com",
    "tennis.com",
    "tennismajors.com",
    "usopen.org",
    "wimbledon.com",
]

EXPANDED_SOURCE_DOMAINS = PRIMARY_SOURCE_DOMAINS + [
    "cbc.ca",
    "sportskeeda.com",
    "tennisworldusa.org",
    "thesportsrush.com",
    "yahoo.com",
]

USER_LOCATION = {
    "type": "approximate",
    "country": "US",
    "timezone": "America/New_York",
}


class InjuryRecord(BaseModel):
    injury_name: str = Field(description="Short label for the injury, for example right hip injury.")
    body_location: str = Field(description="Most specific body location reported in public sources.")
    body_half: Literal["top", "bottom", "unknown"]
    joint_injury: bool
    acute_or_chronic: Literal["acute", "chronic", "unknown"]
    injury_date: Optional[str] = Field(
        default=None,
        description="Date of injury in YYYY-MM-DD if known, otherwise YYYY-MM or YYYY, or null.",
    )
    tennis_related: Literal["yes", "no"]
    estimated_recovery_time: str = Field(
        description=(
            "Typical estimated recovery time for this injury type. "
            "If it is not clearly inferable, use 'unknown'."
        )
    )
    retired_during_match_due_to_injury: Literal["yes", "no"]
    source_urls: list[str] = Field(description="Public source URLs used for this injury record.")


class PlayerInjuryReport(BaseModel):
    player_name: str
    injuries: list[InjuryRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured injury reports for ATP players using OpenAI web search."
    )
    parser.add_argument(
        "--mode",
        choices=["top-ranked", "unique-match-players"],
        default="top-ranked",
        help="How to choose players when --player is not provided.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top-ranked players to analyze when --player is not provided.",
    )
    parser.add_argument(
        "--player",
        action="append",
        dest="players",
        help="Player name to analyze. Repeat the flag to analyze multiple players.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the JSON output file for single-run modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for batched top_XXX_injury_report.json files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of players per batch file in unique-match-players mode.",
    )
    parser.add_argument(
        "--match-glob",
        action="append",
        dest="match_globs",
        help=(
            "Glob for ATP match CSV files when using unique-match-players mode. "
            f"Defaults to {DEFAULT_MATCH_GLOB!r}."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unique-match-players mode from the highest existing top_XXX_injury_report.json batch.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model to use for structured web-backed extraction.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        help="Optional cap on the number of selected players, useful for testing or partial runs.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Initial max output token budget for each player request.",
    )
    return parser.parse_args()


def load_client() -> OpenAI:
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)


def format_compact_date(value: int | str) -> str:
    value = str(value)
    if len(value) == 8:
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def top_players_from_rankings(
    rankings_file: Path = DEFAULT_RANKINGS_FILE,
    players_file: Path = DEFAULT_PLAYERS_FILE,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[str, list[str]]:
    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_file}")
    if not players_file.exists():
        raise FileNotFoundError(f"Players file not found: {players_file}")

    rankings = pd.read_csv(rankings_file)
    players = pd.read_csv(players_file, low_memory=False)

    latest_ranking_date = int(rankings["ranking_date"].max())
    current_rankings = (
        rankings[rankings["ranking_date"] == latest_ranking_date]
        .sort_values("rank")
        .head(top_n)
    )
    merged = current_rankings.merge(
        players[["player_id", "name_first", "name_last"]],
        left_on="player",
        right_on="player_id",
        how="left",
    )
    merged["player_name"] = (
        merged["name_first"].fillna("").astype(str).str.strip()
        + " "
        + merged["name_last"].fillna("").astype(str).str.strip()
    ).str.strip()
    player_names = [name for name in merged["player_name"].tolist() if name]
    return format_compact_date(latest_ranking_date), player_names


def unique_players_from_match_files(match_globs: list[str]) -> tuple[list[str], list[str]]:
    ordered_players: list[str] = []
    seen_players: set[str] = set()
    matched_files: list[str] = []
    seen_files: set[str] = set()

    for pattern in match_globs:
        for path in sorted(BASE_DIR.glob(pattern)):
            path_str = str(path)
            if path_str in seen_files:
                continue
            seen_files.add(path_str)
            matched_files.append(path_str)

            df = pd.read_csv(path, usecols=["winner_name", "loser_name"], low_memory=False)
            for column in ("winner_name", "loser_name"):
                for raw_name in df[column].dropna().astype(str):
                    player_name = raw_name.strip()
                    if not player_name or player_name in seen_players:
                        continue
                    seen_players.add(player_name)
                    ordered_players.append(player_name)

    if not matched_files:
        raise FileNotFoundError(f"No match files found for patterns: {match_globs}")

    return ordered_players, matched_files


def normalize_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url:
            continue
        normalized = url.split("#", 1)[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def fallback_source_urls(response) -> list[str]:
    urls: list[str] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        if not action or not getattr(action, "sources", None):
            continue
        for source in action.sources:
            if getattr(source, "type", None) == "url" and getattr(source, "url", None):
                urls.append(source.url)
    return normalize_urls(urls)


def build_prompt(player_name: str) -> str:
    return (
        f"Find publicly reported injuries for ATP player {player_name}. "
        "Return only injuries that are supported by public web sources. "
        "Ignore rumors, fantasy projections, betting blurbs, and duplicate stories. "
        "Search for direct injury coverage and match reports using terms such as injury, withdrew, retired hurt, "
        "physical issue, and medical timeout. Focus on reporting from 2021 onward. "
        "For each injury, include a short injury name, the most specific body location available, "
        "whether it belongs to the top or bottom half of the body, whether it is a joint injury, "
        "whether it is acute or chronic, the injury date, whether it was tennis related, "
        "the estimated typical recovery time for that injury type, whether the player retired during "
        "the match due to that injury (only mark yes when the injury was tennis related and the public "
        "source explicitly supports it), and the public source URLs used. "
        "Use 'unknown' when a category cannot be supported. "
        "Return an empty injuries list only when you cannot find any supported public injury reports in the allowed domains."
    )


def domains_for_attempt(attempt: int) -> list[str]:
    if attempt >= MAX_RETRIES:
        return EXPANDED_SOURCE_DOMAINS
    return PRIMARY_SOURCE_DOMAINS


def fetch_player_report(
    client: OpenAI,
    player_name: str,
    model: str,
    max_output_tokens: int,
    retry_on_empty: bool = True,
    max_tool_calls: int = 2,
) -> PlayerInjuryReport:
    token_budget = max_output_tokens
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        search_context_size = "low" if attempt == 1 else "medium"
        allowed_domains = domains_for_attempt(attempt)
        try:
            response = client.responses.parse(
                model=model,
                input=build_prompt(player_name),
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": allowed_domains},
                        "search_context_size": search_context_size,
                        "user_location": USER_LOCATION,
                    }
                ],
                include=["web_search_call.action.sources"],
                text_format=PlayerInjuryReport,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                max_tool_calls=max_tool_calls,
                parallel_tool_calls=False,
                max_output_tokens=token_budget,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)
            continue

        parsed = response.output_parsed
        if parsed is None:
            last_error = (
                "OpenAI returned no parsed report "
                f"(status={response.status}, incomplete={response.incomplete_details})"
            )
            if (
                response.incomplete_details
                and getattr(response.incomplete_details, "reason", None) == "max_output_tokens"
                and attempt < MAX_RETRIES
            ):
                token_budget += 1000
                time.sleep(2 ** attempt)
                continue
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)
            continue

        fallback_urls = fallback_source_urls(response)
        for injury in parsed.injuries:
            injury.source_urls = normalize_urls(injury.source_urls or fallback_urls)
            if injury.tennis_related == "no":
                injury.retired_during_match_due_to_injury = "no"
        parsed.player_name = player_name

        if not parsed.injuries and retry_on_empty and attempt < MAX_RETRIES:
            last_error = f"No supported injuries were returned for {player_name} on attempt {attempt}."
            token_budget += 500
            time.sleep(2 ** attempt)
            continue

        return parsed

    raise RuntimeError(last_error or f"Failed to generate a report for {player_name}.")


def build_output_payload(
    reports_by_player: dict[str, dict],
    model: str,
    selection_mode: str,
    players: list[str],
    ranking_date: Optional[str] = None,
    input_match_files: Optional[list[str]] = None,
    player_start_index: int = 1,
    player_end_index: Optional[int] = None,
    total_players_in_scope: Optional[int] = None,
    batch_target_end_index: Optional[int] = None,
    batch_complete: bool = True,
) -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "selection_mode": selection_mode,
        "ranking_date": ranking_date,
        "player_count": len(players),
        "player_start_index": player_start_index,
        "player_end_index": player_end_index if player_end_index is not None else len(players),
        "batch_target_end_index": batch_target_end_index,
        "batch_complete": batch_complete,
        "total_players_in_scope": total_players_in_scope if total_players_in_scope is not None else len(players),
        "input_match_files": input_match_files or [],
        "source_policy": {
            "primary_allowed_domains": PRIMARY_SOURCE_DOMAINS,
            "expanded_allowed_domains": EXPANDED_SOURCE_DOMAINS,
            "user_location": USER_LOCATION,
        },
        "players": reports_by_player,
    }


def batch_output_path(output_dir: Path, player_end_index: int) -> Path:
    return output_dir / f"top_{player_end_index}_injury_report.json"


def partial_batch_output_path(output_dir: Path, target_batch_end_index: int) -> Path:
    return output_dir / f"top_{target_batch_end_index}_injury_report.partial.json"


def highest_completed_batch(output_dir: Path) -> int:
    highest = 0
    for path in output_dir.glob("top_*_injury_report.json"):
        stem = path.stem
        if not stem.startswith("top_") or not stem.endswith("_injury_report"):
            continue
        middle = stem[len("top_") : -len("_injury_report")]
        if middle.isdigit():
            highest = max(highest, int(middle))
    return highest


def latest_partial_batch(output_dir: Path) -> tuple[int, Optional[Path], Optional[dict]]:
    highest = 0
    best_path: Optional[Path] = None
    best_payload: Optional[dict] = None

    for path in output_dir.glob("top_*_injury_report.partial.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        player_end_index = int(payload.get("player_end_index") or 0)
        if player_end_index > highest:
            highest = player_end_index
            best_path = path
            best_payload = payload

    return highest, best_path, best_payload


def describe_players(players: list[str], preview_count: int = 10) -> str:
    if len(players) <= preview_count:
        return str(players)
    preview = ", ".join(players[:preview_count])
    return f"[{preview}, ...] ({len(players)} total)"


def main() -> None:
    args = parse_args()
    client = load_client()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.players:
        selection_mode = "explicit-players"
        ranking_date = None
        players = args.players
        input_match_files: list[str] = []
    elif args.mode == "unique-match-players":
        selection_mode = "unique-match-players"
        ranking_date = None
        match_globs = args.match_globs or [DEFAULT_MATCH_GLOB]
        players, input_match_files = unique_players_from_match_files(match_globs)
    else:
        selection_mode = "top-ranked"
        ranking_date, players = top_players_from_rankings(top_n=args.top_n)
        input_match_files = []

    if args.max_players is not None:
        players = players[: args.max_players]

    print(f"Generating injury reports for {len(players)} players: {describe_players(players)}")
    if ranking_date:
        print(f"Using ATP rankings dated {ranking_date}")

    if selection_mode == "unique-match-players":
        completed_index = highest_completed_batch(args.output_dir) if args.resume else 0
        partial_index = 0
        partial_path: Optional[Path] = None
        partial_payload: Optional[dict] = None
        if args.resume:
            partial_index, partial_path, partial_payload = latest_partial_batch(args.output_dir)

        resume_index = completed_index
        initial_batch_start = completed_index
        if partial_index > completed_index and partial_payload is not None:
            resume_index = partial_index
            initial_batch_start = max(0, ((partial_index - 1) // args.batch_size) * args.batch_size)
            print(f"Resuming unique-match-players mode from partial batch at player index {resume_index + 1}")
        elif resume_index:
            print(f"Resuming unique-match-players mode from player index {resume_index + 1}")

        total_players = len(players)
        for batch_start in range(initial_batch_start, total_players, args.batch_size):
            if batch_start < completed_index:
                continue

            batch_players = players[batch_start : batch_start + args.batch_size]
            batch_reports: dict[str, dict] = {}
            batch_end = batch_start + len(batch_players)
            current_partial_path = partial_batch_output_path(args.output_dir, batch_end)
            processed_in_batch = 0

            if (
                partial_payload is not None
                and partial_path == current_partial_path
                and batch_start < partial_index <= batch_end
            ):
                loaded_reports = partial_payload.get("players", {})
                if isinstance(loaded_reports, dict):
                    batch_reports = loaded_reports
                    processed_in_batch = len(batch_reports)
                    print(
                        f"Loaded partial batch progress for {processed_in_batch} players "
                        f"in batch {batch_start + 1}-{batch_end}"
                    )

            print(
                f"Starting batch {batch_start + 1}-{batch_end} of {total_players} "
                f"into {batch_output_path(args.output_dir, batch_end).name}"
            )

            for offset, player_name in enumerate(batch_players[processed_in_batch:], start=processed_in_batch + 1):
                absolute_index = batch_start + offset
                print(f"[{absolute_index}/{total_players}] Processing {player_name}")
                try:
                    report = fetch_player_report(
                        client=client,
                        player_name=player_name,
                        model=args.model,
                        max_output_tokens=args.max_output_tokens,
                        retry_on_empty=False,
                        max_tool_calls=1,
                    )
                    batch_reports[player_name] = report.model_dump()
                    print(f"Saved structured report for {player_name} with {len(report.injuries)} injuries")
                except Exception as exc:
                    batch_reports[player_name] = {
                        "player_name": player_name,
                        "injuries": [],
                        "error": str(exc),
                    }
                    print(f"Failed to build report for {player_name}: {exc}")

                partial_payload = build_output_payload(
                    reports_by_player=batch_reports,
                    model=args.model,
                    selection_mode=selection_mode,
                    ranking_date=ranking_date,
                    players=batch_players[: len(batch_reports)],
                    input_match_files=input_match_files,
                    player_start_index=batch_start + 1,
                    player_end_index=batch_start + len(batch_reports),
                    total_players_in_scope=total_players,
                    batch_target_end_index=batch_end,
                    batch_complete=False,
                )
                current_partial_path.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")

            output_payload = build_output_payload(
                reports_by_player=batch_reports,
                model=args.model,
                selection_mode=selection_mode,
                ranking_date=ranking_date,
                players=batch_players,
                input_match_files=input_match_files,
                player_start_index=batch_start + 1,
                player_end_index=batch_end,
                total_players_in_scope=total_players,
                batch_target_end_index=batch_end,
                batch_complete=True,
            )

            output_path = batch_output_path(args.output_dir, batch_end)
            output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
            if current_partial_path.exists():
                current_partial_path.unlink()
            print(f"Wrote batch injury report JSON to {output_path}")
        return

    reports_by_player: dict[str, dict] = {}
    for index, player_name in enumerate(players, start=1):
        print(f"[{index}/{len(players)}] Processing {player_name}")
        try:
            report = fetch_player_report(
                client=client,
                player_name=player_name,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                retry_on_empty=True,
                max_tool_calls=2,
            )
            reports_by_player[player_name] = report.model_dump()
            print(f"Saved structured report for {player_name} with {len(report.injuries)} injuries")
        except Exception as exc:
            reports_by_player[player_name] = {
                "player_name": player_name,
                "injuries": [],
                "error": str(exc),
            }
            print(f"Failed to build report for {player_name}: {exc}")

    output_payload = build_output_payload(
        reports_by_player=reports_by_player,
        model=args.model,
        selection_mode=selection_mode,
        ranking_date=ranking_date,
        players=players,
        input_match_files=input_match_files,
        batch_complete=True,
    )

    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote injury report JSON to {args.output}")


if __name__ == "__main__":
    main()
