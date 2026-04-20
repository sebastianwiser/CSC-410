.bail on

-- SQLite pipeline to collapse the multi-file tennis inputs into one CSV.
--
-- Run from CSC410Pres/processing:
--   sqlite3 processing.sqlite < build_single_input.sql
--
-- Output:
--   ./output/consolidated_match_input.csv
--
-- This leaves the model scripts untouched. It only builds a single enriched
-- match-level CSV from the existing processed, initial, serve, playstyle, and
-- injury data sources.

PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
PRAGMA temp_store = MEMORY;

DROP TABLE IF EXISTS raw_processed_source;
CREATE TABLE raw_processed_source (
    tournament TEXT,
    match_date TEXT,
    series TEXT,
    court TEXT,
    surface TEXT,
    round TEXT,
    best_of TEXT,
    player_1 TEXT,
    player_2 TEXT,
    winner TEXT,
    rank_1 TEXT,
    rank_2 TEXT,
    pts_1 TEXT,
    pts_2 TEXT,
    odd_1 TEXT,
    odd_2 TEXT,
    score TEXT
);

.import --csv --skip 1 ../data/processed/atp_tennis.csv raw_processed_source

DROP TABLE IF EXISTS raw_initial_source;
CREATE TABLE raw_initial_source (
    tourney_id TEXT,
    tourney_name TEXT,
    surface TEXT,
    draw_size TEXT,
    tourney_level TEXT,
    tourney_date TEXT,
    match_num TEXT,
    winner_id TEXT,
    winner_seed TEXT,
    winner_entry TEXT,
    winner_name TEXT,
    winner_hand TEXT,
    winner_ht TEXT,
    winner_ioc TEXT,
    winner_age TEXT,
    loser_id TEXT,
    loser_seed TEXT,
    loser_entry TEXT,
    loser_name TEXT,
    loser_hand TEXT,
    loser_ht TEXT,
    loser_ioc TEXT,
    loser_age TEXT,
    score TEXT,
    best_of TEXT,
    round TEXT,
    minutes TEXT,
    w_ace TEXT,
    w_df TEXT,
    w_svpt TEXT,
    w_1stin TEXT,
    w_1stwon TEXT,
    w_2ndwon TEXT,
    w_svgms TEXT,
    w_bpsaved TEXT,
    w_bpfaced TEXT,
    l_ace TEXT,
    l_df TEXT,
    l_svpt TEXT,
    l_1stin TEXT,
    l_1stwon TEXT,
    l_2ndwon TEXT,
    l_svgms TEXT,
    l_bpsaved TEXT,
    l_bpfaced TEXT,
    winner_rank TEXT,
    winner_rank_points TEXT,
    loser_rank TEXT,
    loser_rank_points TEXT
);

.import --csv --skip 1 ../data/initial/atp_matches_2014.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2015.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2016.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2017.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2018.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2019.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2020.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2021.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2022.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2023.csv raw_initial_source
.import --csv --skip 1 ../data/initial/atp_matches_2024.csv raw_initial_source

DROP TABLE IF EXISTS raw_surface_results_source;
CREATE TABLE raw_surface_results_source (
    player_name TEXT,
    surface TEXT,
    matches TEXT,
    wins TEXT,
    losses TEXT
);

.import --csv --skip 1 ../data/initial/player_surface_results.csv raw_surface_results_source

DROP TABLE IF EXISTS raw_serve_source;
CREATE TABLE raw_serve_source (
    player_name TEXT,
    first_serve_pct TEXT,
    first_serve_win_pct TEXT,
    second_serve_win_pct TEXT
);

.import --csv --skip 1 ../data/serve_data/gemini_serve_1.csv raw_serve_source
.import --csv --skip 1 ../data/serve_data/gemini_serve_2.csv raw_serve_source
.import --csv --skip 1 ../data/serve_data/gemini_serve_3.csv raw_serve_source

DROP TABLE IF EXISTS consolidated_match_input;

CREATE TABLE consolidated_match_input AS
WITH
processed_matches AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY
                date(match_date),
                tournament,
                round,
                player_1,
                player_2,
                score
        ) AS match_id,
        tournament,
        date(match_date) AS match_date,
        CAST(strftime('%Y', date(match_date)) AS INTEGER) AS match_year,
        LOWER(TRIM(series)) AS series,
        LOWER(TRIM(court)) AS court,
        LOWER(TRIM(surface)) AS surface,
        LOWER(TRIM(round)) AS match_round,
        TRIM(best_of) AS best_of,
        player_1,
        player_2,
        winner,
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(player_1, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS p1_key,
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(player_2, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS p2_key,
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(winner, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS winner_key,
        CASE WHEN TRIM(rank_1) = '-1' OR TRIM(rank_1) = '' THEN NULL ELSE CAST(rank_1 AS REAL) END AS p1_rank,
        CASE WHEN TRIM(rank_2) = '-1' OR TRIM(rank_2) = '' THEN NULL ELSE CAST(rank_2 AS REAL) END AS p2_rank,
        CASE WHEN TRIM(pts_1) = '-1' OR TRIM(pts_1) = '' THEN NULL ELSE CAST(pts_1 AS REAL) END AS p1_pts,
        CASE WHEN TRIM(pts_2) = '-1' OR TRIM(pts_2) = '' THEN NULL ELSE CAST(pts_2 AS REAL) END AS p2_pts,
        CASE WHEN TRIM(odd_1) = '-1' OR TRIM(odd_1) = '' THEN NULL ELSE CAST(odd_1 AS REAL) END AS odd_1,
        CASE WHEN TRIM(odd_2) = '-1' OR TRIM(odd_2) = '' THEN NULL ELSE CAST(odd_2 AS REAL) END AS odd_2,
        score,
        CASE WHEN winner = player_1 THEN 1 ELSE 0 END AS label
    FROM raw_processed_source
    WHERE date(match_date) IS NOT NULL
      AND TRIM(COALESCE(surface, '')) <> ''
      AND TRIM(COALESCE(round, '')) <> ''
      AND TRIM(COALESCE(rank_1, '')) NOT IN ('', '-1')
      AND TRIM(COALESCE(rank_2, '')) NOT IN ('', '-1')
),

initial_matches AS (
    SELECT
        date(
            substr(tourney_date, 1, 4) || '-' ||
            substr(tourney_date, 5, 2) || '-' ||
            substr(tourney_date, 7, 2)
        ) AS match_date,
        CAST(substr(tourney_date, 1, 4) AS INTEGER) AS match_year,
        winner_name,
        loser_name,
        winner_hand,
        loser_hand,
        CAST(NULLIF(winner_ht, '') AS REAL) AS winner_ht,
        CAST(NULLIF(loser_ht, '') AS REAL) AS loser_ht,
        CAST(NULLIF(winner_age, '') AS REAL) AS winner_age,
        CAST(NULLIF(loser_age, '') AS REAL) AS loser_age,
        CAST(NULLIF(w_ace, '') AS REAL) AS w_ace,
        CAST(NULLIF(w_df, '') AS REAL) AS w_df,
        CAST(NULLIF(w_svpt, '') AS REAL) AS w_svpt,
        CAST(NULLIF(w_1stin, '') AS REAL) AS w_1st_in,
        CAST(NULLIF(w_1stwon, '') AS REAL) AS w_1st_won,
        CAST(NULLIF(w_2ndwon, '') AS REAL) AS w_2nd_won,
        CAST(NULLIF(w_bpsaved, '') AS REAL) AS w_bp_saved,
        CAST(NULLIF(w_bpfaced, '') AS REAL) AS w_bp_faced,
        CAST(NULLIF(l_ace, '') AS REAL) AS l_ace,
        CAST(NULLIF(l_df, '') AS REAL) AS l_df,
        CAST(NULLIF(l_svpt, '') AS REAL) AS l_svpt,
        CAST(NULLIF(l_1stin, '') AS REAL) AS l_1st_in,
        CAST(NULLIF(l_1stwon, '') AS REAL) AS l_1st_won,
        CAST(NULLIF(l_2ndwon, '') AS REAL) AS l_2nd_won,
        CAST(NULLIF(l_bpsaved, '') AS REAL) AS l_bp_saved,
        CAST(NULLIF(l_bpfaced, '') AS REAL) AS l_bp_faced
    FROM raw_initial_source
),

initial_player_long AS (
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(winner_name), ' ') > 0
                    THEN substr(TRIM(winner_name), INSTR(TRIM(winner_name), ' ') + 1) || ' ' || substr(TRIM(winner_name), 1, 1)
                ELSE winner_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        TRIM(COALESCE(winner_hand, '')) AS hand,
        winner_ht AS ht,
        CASE
            WHEN match_year IS NOT NULL AND winner_age IS NOT NULL THEN match_year - winner_age
            ELSE NULL
        END AS birth_year
    FROM initial_matches
    UNION ALL
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(loser_name), ' ') > 0
                    THEN substr(TRIM(loser_name), INSTR(TRIM(loser_name), ' ') + 1) || ' ' || substr(TRIM(loser_name), 1, 1)
                ELSE loser_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        TRIM(COALESCE(loser_hand, '')) AS hand,
        loser_ht AS ht,
        CASE
            WHEN match_year IS NOT NULL AND loser_age IS NOT NULL THEN match_year - loser_age
            ELSE NULL
        END AS birth_year
    FROM initial_matches
),

player_hand_counts AS (
    SELECT
        name_key,
        hand,
        COUNT(*) AS hand_count
    FROM initial_player_long
    WHERE name_key <> ''
      AND hand <> ''
    GROUP BY 1, 2
),

player_hand_choice AS (
    SELECT
        name_key,
        hand,
        ROW_NUMBER() OVER (
            PARTITION BY name_key
            ORDER BY hand_count DESC, hand
        ) AS rn
    FROM player_hand_counts
),

player_ht_ranked AS (
    SELECT
        name_key,
        ht,
        ROW_NUMBER() OVER (PARTITION BY name_key ORDER BY ht) AS rn,
        COUNT(*) OVER (PARTITION BY name_key) AS cnt
    FROM initial_player_long
    WHERE name_key <> ''
      AND ht IS NOT NULL
),

player_ht_median AS (
    SELECT
        name_key,
        AVG(ht) AS ht
    FROM player_ht_ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY 1
),

player_birth_ranked AS (
    SELECT
        name_key,
        birth_year,
        ROW_NUMBER() OVER (PARTITION BY name_key ORDER BY birth_year) AS rn,
        COUNT(*) OVER (PARTITION BY name_key) AS cnt
    FROM initial_player_long
    WHERE name_key <> ''
      AND birth_year IS NOT NULL
),

player_birth_median AS (
    SELECT
        name_key,
        AVG(birth_year) AS birth_year
    FROM player_birth_ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY 1
),

player_keys AS (
    SELECT DISTINCT name_key
    FROM initial_player_long
    WHERE name_key <> ''
),

player_attrs AS (
    SELECT
        k.name_key,
        COALESCE(h.hand, 'U') AS hand,
        hm.ht,
        bm.birth_year
    FROM player_keys k
    LEFT JOIN player_hand_choice h
        ON k.name_key = h.name_key
       AND h.rn = 1
    LEFT JOIN player_ht_median hm
        ON k.name_key = hm.name_key
    LEFT JOIN player_birth_median bm
        ON k.name_key = bm.name_key
),

surface_results AS (
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(player_name), ' ') > 0
                    THEN substr(TRIM(player_name), INSTR(TRIM(player_name), ' ') + 1) || ' ' || substr(TRIM(player_name), 1, 1)
                ELSE player_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        LOWER(TRIM(surface)) AS surface,
        COALESCE(CAST(NULLIF(matches, '') AS REAL), COALESCE(CAST(NULLIF(wins, '') AS REAL), 0) + COALESCE(CAST(NULLIF(losses, '') AS REAL), 0)) AS matches,
        COALESCE(CAST(NULLIF(wins, '') AS REAL), 0) AS wins,
        COALESCE(CAST(NULLIF(losses, '') AS REAL), 0) AS losses,
        CASE
            WHEN COALESCE(CAST(NULLIF(matches, '') AS REAL), COALESCE(CAST(NULLIF(wins, '') AS REAL), 0) + COALESCE(CAST(NULLIF(losses, '') AS REAL), 0)) > 0
                THEN COALESCE(CAST(NULLIF(wins, '') AS REAL), 0) /
                     COALESCE(CAST(NULLIF(matches, '') AS REAL), COALESCE(CAST(NULLIF(wins, '') AS REAL), 0) + COALESCE(CAST(NULLIF(losses, '') AS REAL), 0))
            ELSE 0.5
        END AS win_rate
    FROM raw_surface_results_source
),

serve_lookup AS (
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(player_name), ' ') > 0
                    THEN substr(TRIM(player_name), INSTR(TRIM(player_name), ' ') + 1) || ' ' || substr(TRIM(player_name), 1, 1)
                ELSE player_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        AVG(CAST(NULLIF(first_serve_pct, '') AS REAL)) AS first_serve_pct,
        AVG(CAST(NULLIF(first_serve_win_pct, '') AS REAL)) AS first_serve_win_pct,
        AVG(CAST(NULLIF(second_serve_win_pct, '') AS REAL)) AS second_serve_win_pct
    FROM raw_serve_source
    GROUP BY 1
),

playstyle_files AS (
    SELECT CAST(readfile('../data/playstyle_reports/top_100_playstyle_report.json') AS TEXT) AS json_text
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_200_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_300_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_400_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_500_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_600_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_700_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_800_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_900_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_1000_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_1100_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_1200_playstyle_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/playstyle_reports/top_1261_playstyle_report.json') AS TEXT)
),

playstyle_rows AS (
    SELECT
        json_extract(entry.value, '$.player_name') AS player_name,
        COALESCE(NULLIF(TRIM(json_extract(entry.value, '$.play_style')), ''), 'Unknown') AS play_style
    FROM playstyle_files pf
    JOIN json_each(pf.json_text) AS entry
),

playstyle_lookup AS (
    SELECT
        name_key,
        MAX(play_style) AS play_style
    FROM (
        SELECT
            LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(player_name, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
            play_style
        FROM playstyle_rows
        UNION ALL
        SELECT
            LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                CASE
                    WHEN INSTR(TRIM(player_name), ' ') > 0
                        THEN substr(TRIM(player_name), INSTR(TRIM(player_name), ' ') + 1) || ' ' || substr(TRIM(player_name), 1, 1)
                    ELSE player_name
                END,
                '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
            play_style
        FROM playstyle_rows
    )
    WHERE name_key <> ''
    GROUP BY 1
),

injury_files AS (
    SELECT CAST(readfile('../data/injury_reports/injury_reports_top10.json') AS TEXT) AS json_text
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_100_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_200_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_300_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_400_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_500_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_600_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_700_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_800_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_900_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_1000_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_1100_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_1200_injury_report.json') AS TEXT)
    UNION ALL SELECT CAST(readfile('../data/injury_reports/top_1261_injury_report.json') AS TEXT)
),

injury_players AS (
    SELECT DISTINCT
        COALESCE(NULLIF(json_extract(player.value, '$.player_name'), ''), player.key) AS player_name
    FROM injury_files f
    JOIN json_each(json_extract(f.json_text, '$.players')) AS player
),

injury_profile_lookup AS (
    SELECT DISTINCT name_key
    FROM (
        SELECT
            LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(player_name, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key
        FROM injury_players
        UNION ALL
        SELECT
            LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                CASE
                    WHEN INSTR(TRIM(player_name), ' ') > 0
                        THEN substr(TRIM(player_name), INSTR(TRIM(player_name), ' ') + 1) || ' ' || substr(TRIM(player_name), 1, 1)
                    ELSE player_name
                END,
                '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key
        FROM injury_players
    )
    WHERE name_key <> ''
),

injury_events_base AS (
    SELECT
        COALESCE(NULLIF(json_extract(player.value, '$.player_name'), ''), player.key) AS player_name,
        json_extract(injury.value, '$.injury_name') AS injury_name,
        json_extract(injury.value, '$.injury_date') AS injury_date_raw,
        json_extract(injury.value, '$.estimated_recovery_time') AS recovery_text,
        LOWER(TRIM(COALESCE(json_extract(injury.value, '$.acute_or_chronic'), 'unknown'))) AS acute_or_chronic,
        LOWER(TRIM(COALESCE(json_extract(injury.value, '$.retired_during_match_due_to_injury'), 'no'))) AS retired_raw,
        CASE
            WHEN LOWER(COALESCE(json_extract(injury.value, '$.joint_injury'), 'false')) IN ('true', '1') THEN 1
            ELSE 0
        END AS joint_injury
    FROM injury_files f
    JOIN json_each(json_extract(f.json_text, '$.players')) AS player
    JOIN json_each(json_extract(player.value, '$.injuries')) AS injury
),

injury_events_clean AS (
    SELECT DISTINCT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(player_name, '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS canonical_key,
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(player_name), ' ') > 0
                    THEN substr(TRIM(player_name), INSTR(TRIM(player_name), ' ') + 1) || ' ' || substr(TRIM(player_name), 1, 1)
                ELSE player_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS alias_key,
        LOWER(TRIM(COALESCE(injury_name, ''))) AS injury_name_key,
        CASE
            WHEN injury_date_raw IS NULL OR TRIM(LOWER(injury_date_raw)) IN ('', 'unknown') THEN NULL
            WHEN LENGTH(TRIM(injury_date_raw)) = 10 THEN date(TRIM(injury_date_raw))
            WHEN LENGTH(TRIM(injury_date_raw)) = 7 THEN date(TRIM(injury_date_raw) || '-01')
            WHEN LENGTH(TRIM(injury_date_raw)) = 4 THEN date(TRIM(injury_date_raw) || '-01-01')
            ELSE date(TRIM(injury_date_raw))
        END AS injury_date,
        CASE
            WHEN recovery_text IS NULL OR TRIM(LOWER(recovery_text)) IN ('', 'unknown') THEN NULL
            ELSE
                (
                    CASE
                        WHEN INSTR(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), ' to ') > 0
                            THEN (
                                CAST(REPLACE(LOWER(TRIM(recovery_text)), 'about ', '') AS REAL) +
                                CAST(substr(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), INSTR(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), ' to ') + 4) AS REAL)
                            ) / 2.0
                        WHEN INSTR(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), '-') > 0
                            THEN (
                                CAST(REPLACE(LOWER(TRIM(recovery_text)), 'about ', '') AS REAL) +
                                CAST(substr(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), INSTR(REPLACE(LOWER(TRIM(recovery_text)), 'about ', ''), '-') + 1) AS REAL)
                            ) / 2.0
                        WHEN LOWER(TRIM(recovery_text)) LIKE '%same day%' THEN 2.0
                        WHEN LOWER(TRIM(recovery_text)) LIKE '%day%' AND LOWER(TRIM(recovery_text)) LIKE '%week%' THEN 7.0
                        WHEN LOWER(TRIM(recovery_text)) LIKE '%day%' THEN 2.0
                        WHEN LOWER(TRIM(recovery_text)) LIKE '%week%' THEN 14.0
                        ELSE CAST(REPLACE(LOWER(TRIM(recovery_text)), 'about ', '') AS REAL)
                    END
                ) *
                CASE
                    WHEN LOWER(TRIM(recovery_text)) LIKE '%month%' THEN 30.0
                    WHEN LOWER(TRIM(recovery_text)) LIKE '%week%' THEN 7.0
                    ELSE 1.0
                END
        END AS recovery_days,
        joint_injury,
        CASE WHEN acute_or_chronic = 'chronic' THEN 1 ELSE 0 END AS chronic_injury,
        CASE WHEN acute_or_chronic = 'acute' THEN 1 ELSE 0 END AS acute_injury,
        CASE WHEN retired_raw IN ('yes', 'true', '1') THEN 1 ELSE 0 END AS retired_injury
    FROM injury_events_base
),

injury_event_lookup AS (
    SELECT DISTINCT
        name_key,
        injury_date,
        injury_name_key,
        recovery_days,
        CASE
            WHEN recovery_days IS NOT NULL AND injury_date IS NOT NULL
                THEN date(injury_date, '+' || CAST(ROUND(recovery_days) AS INTEGER) || ' days')
            ELSE NULL
        END AS return_date,
        joint_injury,
        chronic_injury,
        acute_injury,
        retired_injury
    FROM (
        SELECT
            canonical_key AS name_key,
            injury_date,
            injury_name_key,
            recovery_days,
            joint_injury,
            chronic_injury,
            acute_injury,
            retired_injury
        FROM injury_events_clean
        UNION ALL
        SELECT
            alias_key AS name_key,
            injury_date,
            injury_name_key,
            recovery_days,
            joint_injury,
            chronic_injury,
            acute_injury,
            retired_injury
        FROM injury_events_clean
    )
    WHERE name_key <> ''
      AND injury_date IS NOT NULL
),

match_players AS (
    SELECT match_id, match_date, p1_key AS player_key, 'p1' AS player_slot
    FROM processed_matches
    UNION ALL
    SELECT match_id, match_date, p2_key AS player_key, 'p2' AS player_slot
    FROM processed_matches
),

injury_match_events AS (
    SELECT
        mp.match_id,
        mp.player_slot,
        CASE WHEN ip.name_key IS NOT NULL THEN 1 ELSE 0 END AS has_report,
        ie.injury_date,
        ie.return_date,
        ie.recovery_days,
        ie.joint_injury,
        ie.chronic_injury,
        ie.retired_injury,
        CASE
            WHEN ie.injury_date IS NOT NULL THEN CAST(julianday(mp.match_date) - julianday(ie.injury_date) AS INTEGER)
            ELSE NULL
        END AS days_since_injury,
        CASE
            WHEN ie.return_date IS NOT NULL THEN CAST(julianday(mp.match_date) - julianday(ie.return_date) AS INTEGER)
            ELSE NULL
        END AS days_since_return
    FROM match_players mp
    LEFT JOIN injury_profile_lookup ip
        ON mp.player_key = ip.name_key
    LEFT JOIN injury_event_lookup ie
        ON mp.player_key = ie.name_key
       AND ie.injury_date < mp.match_date
),

player_injury_snapshot AS (
    SELECT
        match_id,
        player_slot,
        MAX(has_report) AS has_report,
        CASE
            WHEN SUM(CASE WHEN days_since_injury BETWEEN 0 AND 365 THEN 1 ELSE 0 END) > 0 THEN 1
            ELSE 0
        END AS recent_flag,
        CAST(SUM(CASE WHEN days_since_injury BETWEEN 0 AND 365 THEN 1 ELSE 0 END) AS REAL) AS count_365,
        CAST(COALESCE(MIN(CASE WHEN injury_date IS NOT NULL THEN days_since_injury END), 730) AS REAL) AS days_since_last,
        MAX(CASE WHEN days_since_injury BETWEEN 0 AND 365 AND joint_injury = 1 THEN 1 ELSE 0 END) AS joint_recent,
        MAX(CASE WHEN days_since_injury BETWEEN 0 AND 365 AND chronic_injury = 1 THEN 1 ELSE 0 END) AS chronic_recent,
        MAX(CASE WHEN days_since_injury BETWEEN 0 AND 365 AND retired_injury = 1 THEN 1 ELSE 0 END) AS retired_recent,
        MAX(CASE WHEN days_since_injury BETWEEN 0 AND 365 AND recovery_days IS NOT NULL THEN recovery_days END) AS recovery_days_recent,
        MAX(CASE WHEN days_since_return BETWEEN 0 AND 90 THEN 1 ELSE 0 END) AS recent_return
    FROM injury_match_events
    GROUP BY 1, 2
),

injury_snapshots AS (
    SELECT
        match_id,
        MAX(CASE WHEN player_slot = 'p1' THEN has_report END) AS p1_has_injury_report,
        MAX(CASE WHEN player_slot = 'p2' THEN has_report END) AS p2_has_injury_report,
        MAX(CASE WHEN player_slot = 'p1' THEN recent_flag END) AS p1_inj_recent_flag,
        MAX(CASE WHEN player_slot = 'p2' THEN recent_flag END) AS p2_inj_recent_flag,
        MAX(CASE WHEN player_slot = 'p1' THEN count_365 END) AS p1_inj_count_365,
        MAX(CASE WHEN player_slot = 'p2' THEN count_365 END) AS p2_inj_count_365,
        MAX(CASE WHEN player_slot = 'p1' THEN days_since_last END) AS p1_inj_days_since_last,
        MAX(CASE WHEN player_slot = 'p2' THEN days_since_last END) AS p2_inj_days_since_last,
        MAX(CASE WHEN player_slot = 'p1' THEN joint_recent END) AS p1_inj_joint_recent,
        MAX(CASE WHEN player_slot = 'p2' THEN joint_recent END) AS p2_inj_joint_recent,
        MAX(CASE WHEN player_slot = 'p1' THEN chronic_recent END) AS p1_inj_chronic_recent,
        MAX(CASE WHEN player_slot = 'p2' THEN chronic_recent END) AS p2_inj_chronic_recent,
        MAX(CASE WHEN player_slot = 'p1' THEN retired_recent END) AS p1_inj_retired_recent,
        MAX(CASE WHEN player_slot = 'p2' THEN retired_recent END) AS p2_inj_retired_recent,
        MAX(CASE WHEN player_slot = 'p1' THEN recovery_days_recent END) AS p1_inj_recovery_days_recent,
        MAX(CASE WHEN player_slot = 'p2' THEN recovery_days_recent END) AS p2_inj_recovery_days_recent,
        MAX(CASE WHEN player_slot = 'p1' THEN recent_return END) AS p1_inj_recent_return,
        MAX(CASE WHEN player_slot = 'p2' THEN recent_return END) AS p2_inj_recent_return
    FROM player_injury_snapshot
    GROUP BY 1
),

player_profile_rows AS (
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(winner_name), ' ') > 0
                    THEN substr(TRIM(winner_name), INSTR(TRIM(winner_name), ' ') + 1) || ' ' || substr(TRIM(winner_name), 1, 1)
                ELSE winner_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        CASE WHEN w_svpt > 0 THEN w_ace / w_svpt END AS ace_rate,
        CASE WHEN w_svpt > 0 THEN w_df / w_svpt END AS df_rate,
        CASE WHEN w_svpt > 0 THEN w_1st_in / w_svpt END AS first_in_rate,
        CASE WHEN w_1st_in > 0 THEN w_1st_won / w_1st_in END AS first_win_rate,
        CASE WHEN (w_svpt - w_1st_in) > 0 THEN w_2nd_won / (w_svpt - w_1st_in) END AS second_win_rate,
        CASE WHEN w_bp_faced > 0 THEN w_bp_saved / w_bp_faced END AS bp_save_rate,
        CASE WHEN l_svpt > 0 THEN (l_svpt - l_1st_won - l_2nd_won) / l_svpt END AS return_win_rate
    FROM initial_matches
    UNION ALL
    SELECT
        LOWER(TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
            CASE
                WHEN INSTR(TRIM(loser_name), ' ') > 0
                    THEN substr(TRIM(loser_name), INSTR(TRIM(loser_name), ' ') + 1) || ' ' || substr(TRIM(loser_name), 1, 1)
                ELSE loser_name
            END,
            '.', ''), ',', ''), '-', ' '), '''', ''), '/', ' '), '(', ' '), ')', ' '), '  ', ' '), '  ', ' '))) AS name_key,
        CASE WHEN l_svpt > 0 THEN l_ace / l_svpt END AS ace_rate,
        CASE WHEN l_svpt > 0 THEN l_df / l_svpt END AS df_rate,
        CASE WHEN l_svpt > 0 THEN l_1st_in / l_svpt END AS first_in_rate,
        CASE WHEN l_1st_in > 0 THEN l_1st_won / l_1st_in END AS first_win_rate,
        CASE WHEN (l_svpt - l_1st_in) > 0 THEN l_2nd_won / (l_svpt - l_1st_in) END AS second_win_rate,
        CASE WHEN l_bp_faced > 0 THEN l_bp_saved / l_bp_faced END AS bp_save_rate,
        CASE WHEN w_svpt > 0 THEN (w_svpt - w_1st_won - w_2nd_won) / w_svpt END AS return_win_rate
    FROM initial_matches
),

player_profiles AS (
    SELECT
        name_key,
        AVG(ace_rate) AS ace_rate,
        AVG(df_rate) AS df_rate,
        AVG(first_in_rate) AS first_in_rate,
        AVG(first_win_rate) AS first_win_rate,
        AVG(second_win_rate) AS second_win_rate,
        AVG(bp_save_rate) AS bp_save_rate,
        AVG(return_win_rate) AS return_win_rate,
        COUNT(*) AS profile_match_rows
    FROM player_profile_rows
    WHERE name_key <> ''
    GROUP BY 1
)

SELECT
    pm.match_id,
    pm.tournament,
    pm.match_date,
    pm.match_year AS year,
    pm.series,
    pm.court,
    pm.surface,
    pm.match_round AS round,
    pm.best_of,
    pm.player_1,
    pm.player_2,
    pm.winner,
    pm.p1_key,
    pm.p2_key,
    pm.winner_key,
    pm.label,
    pm.p1_rank,
    pm.p2_rank,
    pm.p1_pts,
    pm.p2_pts,
    pm.odd_1,
    pm.odd_2,
    pm.score,

    COALESCE(pa1.hand, 'U') AS p1_hand,
    COALESCE(pa2.hand, 'U') AS p2_hand,
    pa1.ht AS p1_ht,
    pa2.ht AS p2_ht,
    pa1.birth_year AS p1_birth_year,
    pa2.birth_year AS p2_birth_year,
    pm.match_year - pa1.birth_year AS p1_age,
    pm.match_year - pa2.birth_year AS p2_age,

    sr1.matches AS p1_surface_matches,
    sr2.matches AS p2_surface_matches,
    sr1.wins AS p1_surface_wins,
    sr2.wins AS p2_surface_wins,
    sr1.losses AS p1_surface_losses,
    sr2.losses AS p2_surface_losses,
    sr1.win_rate AS p1_swr,
    sr2.win_rate AS p2_swr,

    sv1.first_serve_pct AS p1_fsp,
    sv2.first_serve_pct AS p2_fsp,
    sv1.first_serve_win_pct AS p1_fswp,
    sv2.first_serve_win_pct AS p2_fswp,
    sv1.second_serve_win_pct AS p1_sswp,
    sv2.second_serve_win_pct AS p2_sswp,
    CASE
        WHEN sv1.first_serve_pct IS NOT NULL
         AND sv1.first_serve_win_pct IS NOT NULL
         AND sv1.second_serve_win_pct IS NOT NULL
        THEN ((sv1.first_serve_pct / 100.0) * (sv1.first_serve_win_pct / 100.0))
           + ((1.0 - (sv1.first_serve_pct / 100.0)) * (sv1.second_serve_win_pct / 100.0))
        ELSE NULL
    END AS p1_serve_dominance,
    CASE
        WHEN sv2.first_serve_pct IS NOT NULL
         AND sv2.first_serve_win_pct IS NOT NULL
         AND sv2.second_serve_win_pct IS NOT NULL
        THEN ((sv2.first_serve_pct / 100.0) * (sv2.first_serve_win_pct / 100.0))
           + ((1.0 - (sv2.first_serve_pct / 100.0)) * (sv2.second_serve_win_pct / 100.0))
        ELSE NULL
    END AS p2_serve_dominance,
    CASE WHEN sv1.first_serve_pct IS NOT NULL THEN 1 ELSE 0 END AS p1_has_serve,
    CASE WHEN sv2.first_serve_pct IS NOT NULL THEN 1 ELSE 0 END AS p2_has_serve,

    COALESCE(ps1.play_style, 'Unknown') AS p1_style,
    COALESCE(ps2.play_style, 'Unknown') AS p2_style,
    COALESCE(ps1.play_style, 'Unknown') || ' vs ' || COALESCE(ps2.play_style, 'Unknown') AS style_matchup,
    CASE
        WHEN COALESCE(ps1.play_style, 'Unknown') <= COALESCE(ps2.play_style, 'Unknown')
            THEN COALESCE(ps1.play_style, 'Unknown') || ' vs ' || COALESCE(ps2.play_style, 'Unknown')
        ELSE COALESCE(ps2.play_style, 'Unknown') || ' vs ' || COALESCE(ps1.play_style, 'Unknown')
    END AS style_pair,
    CASE WHEN COALESCE(ps1.play_style, 'Unknown') = COALESCE(ps2.play_style, 'Unknown') THEN 1 ELSE 0 END AS same_style,
    CASE WHEN COALESCE(ps1.play_style, 'Unknown') <> 'Unknown' THEN 1 ELSE 0 END AS p1_has_style,
    CASE WHEN COALESCE(ps2.play_style, 'Unknown') <> 'Unknown' THEN 1 ELSE 0 END AS p2_has_style,

    pp1.ace_rate AS p1_profile_ace_rate,
    pp2.ace_rate AS p2_profile_ace_rate,
    pp1.df_rate AS p1_profile_df_rate,
    pp2.df_rate AS p2_profile_df_rate,
    pp1.first_in_rate AS p1_profile_first_in_rate,
    pp2.first_in_rate AS p2_profile_first_in_rate,
    pp1.first_win_rate AS p1_profile_first_win_rate,
    pp2.first_win_rate AS p2_profile_first_win_rate,
    pp1.second_win_rate AS p1_profile_second_win_rate,
    pp2.second_win_rate AS p2_profile_second_win_rate,
    pp1.bp_save_rate AS p1_profile_bp_save_rate,
    pp2.bp_save_rate AS p2_profile_bp_save_rate,
    pp1.return_win_rate AS p1_profile_return_win_rate,
    pp2.return_win_rate AS p2_profile_return_win_rate,
    pp1.profile_match_rows AS p1_profile_match_rows,
    pp2.profile_match_rows AS p2_profile_match_rows,

    COALESCE(ins.p1_has_injury_report, 0) AS p1_has_injury_report,
    COALESCE(ins.p2_has_injury_report, 0) AS p2_has_injury_report,
    COALESCE(ins.p1_inj_recent_flag, 0) AS p1_inj_recent_flag,
    COALESCE(ins.p2_inj_recent_flag, 0) AS p2_inj_recent_flag,
    COALESCE(ins.p1_inj_count_365, 0) AS p1_inj_count_365,
    COALESCE(ins.p2_inj_count_365, 0) AS p2_inj_count_365,
    COALESCE(ins.p1_inj_days_since_last, 730) AS p1_inj_days_since_last,
    COALESCE(ins.p2_inj_days_since_last, 730) AS p2_inj_days_since_last,
    COALESCE(ins.p1_inj_joint_recent, 0) AS p1_inj_joint_recent,
    COALESCE(ins.p2_inj_joint_recent, 0) AS p2_inj_joint_recent,
    COALESCE(ins.p1_inj_chronic_recent, 0) AS p1_inj_chronic_recent,
    COALESCE(ins.p2_inj_chronic_recent, 0) AS p2_inj_chronic_recent,
    COALESCE(ins.p1_inj_retired_recent, 0) AS p1_inj_retired_recent,
    COALESCE(ins.p2_inj_retired_recent, 0) AS p2_inj_retired_recent,
    ins.p1_inj_recovery_days_recent,
    ins.p2_inj_recovery_days_recent,
    COALESCE(ins.p1_inj_recent_return, 0) AS p1_inj_recent_return,
    COALESCE(ins.p2_inj_recent_return, 0) AS p2_inj_recent_return,

    pm.p1_rank - pm.p2_rank AS rank_diff,
    COALESCE(pm.p1_pts, 0) - COALESCE(pm.p2_pts, 0) AS pts_diff,
    pm.p1_rank / CASE WHEN pm.p2_rank < 1 THEN 1 ELSE pm.p2_rank END AS rank_ratio,
    ln(1 + pm.p1_rank) - ln(1 + pm.p2_rank) AS log_rank_diff,
    (pm.match_year - pa1.birth_year) - (pm.match_year - pa2.birth_year) AS age_diff,
    pa1.ht - pa2.ht AS ht_diff,
    sr1.win_rate - sr2.win_rate AS surf_win_rate_diff,
    COALESCE(sr1.matches, 0) - COALESCE(sr2.matches, 0) AS surf_exp_diff,
    sv1.first_serve_pct - sv2.first_serve_pct AS first_serve_pct_diff,
    sv1.first_serve_win_pct - sv2.first_serve_win_pct AS first_serve_win_pct_diff,
    sv1.second_serve_win_pct - sv2.second_serve_win_pct AS second_serve_win_pct_diff,
    (
        CASE
            WHEN sv1.first_serve_pct IS NOT NULL
             AND sv1.first_serve_win_pct IS NOT NULL
             AND sv1.second_serve_win_pct IS NOT NULL
            THEN ((sv1.first_serve_pct / 100.0) * (sv1.first_serve_win_pct / 100.0))
               + ((1.0 - (sv1.first_serve_pct / 100.0)) * (sv1.second_serve_win_pct / 100.0))
            ELSE NULL
        END
    ) - (
        CASE
            WHEN sv2.first_serve_pct IS NOT NULL
             AND sv2.first_serve_win_pct IS NOT NULL
             AND sv2.second_serve_win_pct IS NOT NULL
            THEN ((sv2.first_serve_pct / 100.0) * (sv2.first_serve_win_pct / 100.0))
               + ((1.0 - (sv2.first_serve_pct / 100.0)) * (sv2.second_serve_win_pct / 100.0))
            ELSE NULL
        END
    ) AS serve_dominance_diff,
    COALESCE(ins.p1_inj_recent_flag, 0) - COALESCE(ins.p2_inj_recent_flag, 0) AS inj_recent_flag_diff,
    COALESCE(ins.p1_inj_count_365, 0) - COALESCE(ins.p2_inj_count_365, 0) AS inj_count_365_diff,
    COALESCE(ins.p1_inj_days_since_last, 730) - COALESCE(ins.p2_inj_days_since_last, 730) AS inj_days_since_last_diff,
    COALESCE(ins.p1_inj_joint_recent, 0) - COALESCE(ins.p2_inj_joint_recent, 0) AS inj_joint_recent_diff,
    COALESCE(ins.p1_inj_chronic_recent, 0) - COALESCE(ins.p2_inj_chronic_recent, 0) AS inj_chronic_recent_diff,
    COALESCE(ins.p1_inj_retired_recent, 0) - COALESCE(ins.p2_inj_retired_recent, 0) AS inj_retired_recent_diff,
    ins.p1_inj_recovery_days_recent - ins.p2_inj_recovery_days_recent AS inj_recovery_days_diff,
    COALESCE(ins.p1_inj_recent_return, 0) - COALESCE(ins.p2_inj_recent_return, 0) AS inj_recent_return_diff,
    (CASE WHEN sv1.first_serve_pct IS NOT NULL THEN 1 ELSE 0 END)
        + (CASE WHEN sv2.first_serve_pct IS NOT NULL THEN 1 ELSE 0 END) AS serve_coverage,
    (CASE WHEN COALESCE(ps1.play_style, 'Unknown') <> 'Unknown' THEN 1 ELSE 0 END)
        + (CASE WHEN COALESCE(ps2.play_style, 'Unknown') <> 'Unknown' THEN 1 ELSE 0 END) AS playstyle_coverage,
    COALESCE(ins.p1_has_injury_report, 0) + COALESCE(ins.p2_has_injury_report, 0) AS injury_coverage
FROM processed_matches pm
LEFT JOIN player_attrs pa1
    ON pm.p1_key = pa1.name_key
LEFT JOIN player_attrs pa2
    ON pm.p2_key = pa2.name_key
LEFT JOIN surface_results sr1
    ON pm.p1_key = sr1.name_key
   AND pm.surface = sr1.surface
LEFT JOIN surface_results sr2
    ON pm.p2_key = sr2.name_key
   AND pm.surface = sr2.surface
LEFT JOIN serve_lookup sv1
    ON pm.p1_key = sv1.name_key
LEFT JOIN serve_lookup sv2
    ON pm.p2_key = sv2.name_key
LEFT JOIN playstyle_lookup ps1
    ON pm.p1_key = ps1.name_key
LEFT JOIN playstyle_lookup ps2
    ON pm.p2_key = ps2.name_key
LEFT JOIN player_profiles pp1
    ON pm.p1_key = pp1.name_key
LEFT JOIN player_profiles pp2
    ON pm.p2_key = pp2.name_key
LEFT JOIN injury_snapshots ins
    ON pm.match_id = ins.match_id
ORDER BY pm.match_date, pm.match_id;

.headers on
.mode csv
.once ./output/consolidated_match_input.csv
SELECT * FROM consolidated_match_input;
