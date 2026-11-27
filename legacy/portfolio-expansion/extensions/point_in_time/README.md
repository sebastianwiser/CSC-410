# Historical serve-feature extension

`features.py` pools valid serve-count numerators and denominators in a bounded
per-player trailing match window. Dates are processed in batches and histories
update only after the entire date group. Features with no history remain missing
and imputation is fitted on training data only.

The same safe base XGBoost configuration is evaluated with and without historical
serve differences. Tournament dates are event starts rather than true match
timestamps; excluding same-date history is conservative. Earlier test outcomes
may update later test histories, but the final model parameters remain frozen.
Ranks, ages and event context are treated as supplied pre-match metadata, whose
publication dates are not independently established by the source files.

The source-date upper bounds are saved in results/point_in_time_provenance.csv.
Tests additionally perturb same-date/future outcomes and truncate future rows.
