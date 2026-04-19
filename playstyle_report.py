from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "atp_players_1261.csv"
DEFAULT_OUTPUT_JSON = BASE_DIR / "atp_playstyle_classifications.json"
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_PLAYERS_FILE = BASE_DIR / "tennis_atp" / "atp_players.csv"
DEFAULT_RANKINGS_FILE = BASE_DIR / "tennis_atp" / "atp_rankings_current.csv"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 500
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3

USER_LOCATION = {
    "type": "approximate",
    "country": "US",
    "timezone": "America/New_York",
}

CATEGORY_NAMES = [
    "Aggressive Baseliner",
    "Counterpuncher",
    "All-Court Player",
    "Serve-and-Volleyer",
]


class PlayerPlaystyleLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_name: str
    play_style: Literal[
        "Aggressive Baseliner",
        "Counterpuncher",
        "All-Court Player",
        "Serve-and-Volleyer",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify ATP players into one dominant play style using OpenAI web search."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV file containing ATP player names.",
    )
    parser.add_argument(
        "--player-column",
        help="Column containing player names. If omitted, the script auto-detects a likely player-name column.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Final JSON output path for single-file mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for batch files in batched mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["single-file", "batched"],
        default="batched",
        help="Whether to write one JSON file or 100-player batch files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Players per batch file in batched mode.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model to use for classification.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        help="Optional cap on rows processed, useful for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial or final JSON output file when possible.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Initial max output token budget for each player request.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=1,
        help="Maximum web-search tool calls per player request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    return parser.parse_args()


def load_client() -> OpenAI:
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)


def normalize_name(name: str) -> str:
    return " ".join(name.lower().split())


def normalize_text(value: str, fallback: str = "unknown") -> str:
    compact = " ".join((value or "").split())
    return compact or fallback


def handedness_label(raw_hand: str) -> str:
    hand = (raw_hand or "").strip().upper()
    if hand == "R":
        return "right-handed"
    if hand == "L":
        return "left-handed"
    return "unknown"


def detect_player_column(fieldnames: list[str]) -> str:
    preferred = [
        "player_name",
        "player",
        "name",
        "winner_name",
        "loser_name",
        "Player",
        "Player_1",
    ]
    for candidate in preferred:
        if candidate in fieldnames:
            return candidate

    if len(fieldnames) == 1:
        return fieldnames[0]

    raise ValueError(
        f"Could not detect a player-name column from {fieldnames}. Use --player-column explicitly."
    )


def load_players(input_csv: Path, player_column: Optional[str]) -> tuple[str, list[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {input_csv}")

        selected_column = player_column or detect_player_column(reader.fieldnames)
        if selected_column not in reader.fieldnames:
            raise ValueError(
                f"Column {selected_column!r} not found in {input_csv}. Available columns: {reader.fieldnames}"
            )

        players: list[str] = []
        for row in reader:
            player_name = normalize_text(row.get(selected_column, ""), fallback="")
            if player_name:
                players.append(player_name)

    if not players:
        raise ValueError(f"No player names found in column {selected_column!r} of {input_csv}")

    return selected_column, players


def load_handedness_lookup(players_file: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not players_file.exists():
        return lookup

    with players_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = normalize_text(f"{row.get('name_first', '')} {row.get('name_last', '')}", fallback="")
            if not name:
                continue
            lookup[normalize_name(name)] = handedness_label(row.get("hand", ""))

    return lookup


def load_current_ranked_names(rankings_file: Path, players_file: Path) -> set[str]:
    if not rankings_file.exists() or not players_file.exists():
        return set()

    latest_ranking_date: Optional[int] = None
    current_player_ids: set[str] = set()

    with rankings_file.open(encoding="utf-8", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))

    for row in ranking_rows:
        raw_date = (row.get("ranking_date") or "").strip()
        if not raw_date:
            continue
        ranking_date = int(raw_date)
        if latest_ranking_date is None or ranking_date > latest_ranking_date:
            latest_ranking_date = ranking_date

    if latest_ranking_date is None:
        return set()

    for row in ranking_rows:
        raw_date = (row.get("ranking_date") or "").strip()
        player_id = (row.get("player") or "").strip()
        if raw_date and player_id and int(raw_date) == latest_ranking_date:
            current_player_ids.add(player_id)

    current_names: set[str] = set()
    with players_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            player_id = (row.get("player_id") or "").strip()
            if player_id not in current_player_ids:
                continue
            player_name = normalize_text(
                f"{row.get('name_first', '')} {row.get('name_last', '')}",
                fallback="",
            )
            if player_name:
                current_names.add(normalize_name(player_name))

    return current_names


def player_context(
    player_name: str,
    handedness_lookup: dict[str, str],
    current_ranked_names: set[str],
) -> dict[str, str]:
    normalized = normalize_name(player_name)
    return {
        "handedness_hint": handedness_lookup.get(normalized, "unknown"),
        "status_hint": "active" if normalized in current_ranked_names else "unknown",
    }


def build_prompt(player_name: str, context: dict[str, str]) -> str:
    return f"""
You are classifying one ATP professional tennis player into exactly one dominant play style.

PLAYER
- player_name: {player_name}

LOCAL HINTS
- handedness_hint: {context['handedness_hint']}
- status_hint: {context['status_hint']}
Use these hints only if they are consistent with credible evidence.

Choose exactly one of these four categories:
1. Aggressive Baseliner
2. Counterpuncher
3. All-Court Player
4. Serve-and-Volleyer

Use the same evaluation framework every time:
- dominant identity matters more than occasional variation
- if balanced and adaptable, prefer All-Court Player
- if baseline-centered and offensive, prefer Aggressive Baseliner
- if baseline-centered and defensive, prefer Counterpuncher
- only choose Serve-and-Volleyer when it is clearly a defining and frequent pattern

Use reliable match statistics, official or reputable player profiles, and credible tennis analysis when available.
Do not rely on vague reputation, one highlight clip, or one match only.

Think through the classification carefully, but output only a single JSON object with exactly these fields:
{{
  "player_name": "{player_name}",
  "play_style": "Aggressive Baseliner | Counterpuncher | All-Court Player | Serve-and-Volleyer"
}}

Do not include explanations, scores, confidence, citations, markdown, or extra keys.
""".strip()


def partial_output_path(output_json: Path) -> Path:
    return output_json.with_name(f"{output_json.stem}.partial{output_json.suffix}")


def batch_output_path(output_dir: Path, player_end_index: int) -> Path:
    return output_dir / f"top_{player_end_index}_playstyle_report.json"


def partial_batch_output_path(output_dir: Path, target_batch_end_index: int) -> Path:
    return output_dir / f"top_{target_batch_end_index}_playstyle_report.partial.json"


def expected_batch_endings(total_players: int, batch_size: int) -> set[int]:
    return set(range(batch_size, total_players, batch_size)) | {total_players}


def load_reports_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return normalize_existing_reports(payload)


def normalize_existing_reports(payload: object) -> list[dict]:
    if not isinstance(payload, list):
        return []

    normalized_reports: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            return []
        player_name = normalize_text(str(item.get("player_name", "")), fallback="")
        play_style = item.get("play_style") or item.get("final_category")
        play_style = normalize_text(str(play_style or ""), fallback="")
        if not player_name or play_style not in CATEGORY_NAMES:
            return []
        normalized_reports.append({"player_name": player_name, "play_style": play_style})
    return normalized_reports


def load_resume_reports(output_json: Path) -> list[dict]:
    candidates = [partial_output_path(output_json), output_json]
    best_reports: list[dict] = []

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reports = normalize_existing_reports(payload)
        if len(reports) > len(best_reports):
            best_reports = reports

    return best_reports


def highest_completed_batch(output_dir: Path, valid_batch_endings: set[int]) -> int:
    highest = 0
    for path in output_dir.glob("top_*_playstyle_report.json"):
        stem = path.stem
        if not stem.startswith("top_") or not stem.endswith("_playstyle_report"):
            continue
        middle = stem[len("top_") : -len("_playstyle_report")]
        if middle.isdigit():
            player_end_index = int(middle)
            if player_end_index in valid_batch_endings:
                highest = max(highest, player_end_index)
    return highest


def validate_resume_alignment(existing_reports: list[dict], players: list[str]) -> None:
    if len(existing_reports) > len(players):
        raise ValueError(
            "Existing resume file has more rows than the current player selection. "
            "Use a different output file or remove the resume artifacts."
        )

    for index, report in enumerate(existing_reports):
        expected_name = players[index]
        actual_name = normalize_text(str(report.get("player_name", "")), fallback="")
        if actual_name != expected_name:
            raise ValueError(
                "Resume data does not align with the current player order at row "
                f"{index + 1}: expected {expected_name!r}, found {actual_name!r}."
            )


def clean_report(report: PlayerPlaystyleLabel, player_name: str) -> PlayerPlaystyleLabel:
    report.player_name = player_name
    return report


def validate_report(report: PlayerPlaystyleLabel) -> None:
    if report.play_style not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported play_style: {report.play_style}")


def fetch_player_report(
    client: OpenAI,
    player_name: str,
    model: str,
    context: dict[str, str],
    max_output_tokens: int,
    max_tool_calls: int,
) -> PlayerPlaystyleLabel:
    token_budget = max_output_tokens
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        search_context_size = "medium" if attempt == 1 else "high"
        try:
            response = client.responses.parse(
                model=model,
                input=build_prompt(player_name, context),
                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": search_context_size,
                        "user_location": USER_LOCATION,
                    }
                ],
                text_format=PlayerPlaystyleLabel,
                max_tool_calls=max_tool_calls,
                parallel_tool_calls=False,
                max_output_tokens=token_budget,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(2**attempt)
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
                token_budget += 200
                time.sleep(2**attempt)
                continue
            if attempt == MAX_RETRIES:
                break
            time.sleep(2**attempt)
            continue

        try:
            cleaned = clean_report(parsed, player_name=player_name)
            validate_report(cleaned)
        except Exception as exc:
            last_error = str(exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(2**attempt)
            continue

        return cleaned

    raise RuntimeError(last_error or f"Failed to classify {player_name}.")


def save_reports(path: Path, reports: list[dict]) -> None:
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    client = load_client()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_column, players = load_players(args.input_csv, args.player_column)
    if args.max_players is not None:
        players = players[: args.max_players]

    handedness_lookup = load_handedness_lookup(DEFAULT_PLAYERS_FILE)
    current_ranked_names = load_current_ranked_names(DEFAULT_RANKINGS_FILE, DEFAULT_PLAYERS_FILE)

    print(
        f"Loaded {len(players)} ATP players from {args.input_csv.name} using column {selected_column!r}"
    )

    if args.mode == "batched":
        valid_batch_endings = expected_batch_endings(len(players), args.batch_size)
        completed_index = 0
        if args.resume:
            completed_index = highest_completed_batch(args.output_dir, valid_batch_endings)
            if completed_index:
                print(f"Resuming batched mode from player {completed_index + 1}")

        if completed_index >= len(players):
            print(f"All selected players are already classified in {args.output_dir}")
            return

        for batch_start in range(completed_index, len(players), args.batch_size):
            batch_players = players[batch_start : batch_start + args.batch_size]
            batch_end = batch_start + len(batch_players)
            output_path = batch_output_path(args.output_dir, batch_end)
            partial_path = partial_batch_output_path(args.output_dir, batch_end)
            batch_reports: list[dict] = []

            if args.resume:
                batch_reports = load_reports_file(partial_path)
                if batch_reports:
                    validate_resume_alignment(batch_reports, batch_players)
                    print(
                        f"Loaded partial batch progress for {len(batch_reports)} players "
                        f"in batch {batch_start + 1}-{batch_end}"
                    )

            print(
                f"Starting batch {batch_start + 1}-{batch_end} of {len(players)} "
                f"into {output_path.name}"
            )

            for offset, player_name in enumerate(batch_players[len(batch_reports) :], start=len(batch_reports) + 1):
                absolute_index = batch_start + offset
                print(f"[{absolute_index}/{len(players)}] Classifying {player_name}")
                context = player_context(
                    player_name=player_name,
                    handedness_lookup=handedness_lookup,
                    current_ranked_names=current_ranked_names,
                )

                report = fetch_player_report(
                    client=client,
                    player_name=player_name,
                    model=args.model,
                    context=context,
                    max_output_tokens=args.max_output_tokens,
                    max_tool_calls=args.max_tool_calls,
                )
                batch_reports.append(report.model_dump())
                save_reports(partial_path, batch_reports)

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            save_reports(output_path, batch_reports)
            if partial_path.exists():
                partial_path.unlink()
            print(f"Wrote ATP playstyle batch JSON to {output_path}")
        return

    reports: list[dict] = []
    partial_path = partial_output_path(args.output_json)

    if args.resume:
        reports = load_resume_reports(args.output_json)
        if reports:
            validate_resume_alignment(reports, players)
            print(f"Resuming from player {len(reports) + 1} using {len(reports)} existing reports")

    start_index = len(reports)
    if start_index >= len(players):
        save_reports(args.output_json, reports)
        if partial_path.exists():
            partial_path.unlink()
        print(f"All selected players are already classified in {args.output_json}")
        return

    for index, player_name in enumerate(players[start_index:], start=start_index + 1):
        print(f"[{index}/{len(players)}] Classifying {player_name}")
        context = player_context(
            player_name=player_name,
            handedness_lookup=handedness_lookup,
            current_ranked_names=current_ranked_names,
        )

        report = fetch_player_report(
            client=client,
            player_name=player_name,
            model=args.model,
            context=context,
            max_output_tokens=args.max_output_tokens,
            max_tool_calls=args.max_tool_calls,
        )
        reports.append(report.model_dump())
        save_reports(partial_path, reports)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    save_reports(args.output_json, reports)
    if partial_path.exists():
        partial_path.unlink()
    print(f"Wrote ATP playstyle JSON to {args.output_json}")


if __name__ == "__main__":
    main()
