# Supplied serve data audit

gemini files may be LLM-sourced. ATP counts are pooled over the available match files; scope differs from undocumented career estimates.
Missing-stat matches are excluded. Exact full-name joins only; unmatched players are not assumed accurate.

| Stat | Matched rows | MAE (percentage points) | Pearson r | Material fraction |
|---|---:|---:|---:|---:|
| first_serve_pct | 1094 | 4.303340 | 0.217286 | 0.323583 |
| first_serve_win_pct | 1094 | 5.185135 | 0.525471 | 0.360146 |
| second_serve_win_pct | 1094 | 5.354825 | 0.359301 | 0.369287 |

Material means absolute error > 5 percentage points (declared threshold, not learned).
Matched 1094 of 1262 supplied players; player-level any-stat material disagreement fraction: 0.632541.
No inference about deliberate fabrication or individual player health is supported.
