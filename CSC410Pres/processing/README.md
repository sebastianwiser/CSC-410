# Processing

`build_single_input.sql` creates one wide CSV that consolidates the files the models currently read separately.

It is designed for the SQLite CLI. The script uses SQLite import/export dot-commands for CSV files and SQLite JSON functions for the report files.

## What It Builds

Running the script writes:

`output/consolidated_match_input.csv`

Each row is one match from `data/processed/atp_tennis.csv`, enriched with:

- aggregated player attributes from `data/initial/atp_matches_*.csv`
- surface history from `data/initial/player_surface_results.csv`
- serve stats from `data/serve_data/gemini_serve_*.csv`
- playstyles from `data/playstyle_reports/*.json`
- injury snapshots from `data/injury_reports/*.json`
- helper diff columns such as serve, injury, rank, age, and surface deltas
- per-player serve/return profile averages derived from the ATP match files

## Run It

From `CSC410Pres/processing`:

```bash
sqlite3 processing.sqlite < build_single_input.sql
```

## Notes

- The SQL intentionally uses ATP-style abbreviated keys to bridge the name-format mismatch between `atp_tennis.csv` and the full-name sources in `initial/`.
- The model scripts were not changed. This folder only produces the single consolidated input file.
- This export replaces the multi-file static joins. The existing Python models still compute sequential features like Elo, rolling form, and head-to-head history on top of the match timeline.
- The per-player profile columns are included so model-side clustering or further feature engineering can also start from this single file later.
- SQLite has fewer built-in text-normalization helpers than Python, so the name cleaning here is intentionally lightweight and focused on the ATP abbreviation bridge.
- SQLite can build this file, but it is slower than DuckDB on the injury/playstyle enrichment step because the JSON flattening and joins are happening inside SQLite rather than a columnar engine.
