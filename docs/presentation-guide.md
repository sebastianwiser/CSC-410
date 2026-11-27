# Reading the presentation alongside the repository

The [supplied presentation](presentation.pdf) is the starting point. This guide maps its sections to the code and distinguishes the original narrative from the later reproduction work.

| Presentation section | Slides | Supporting files |
|---|---|---|
| Research question and data selection | 3–10 | [Match and supplemental data](../CSC410Pres/data) |
| Logistic baseline | 12–18 | [Baseline script](../CSC410Pres/models/model_1/model_1.py) |
| Improved model and chronological features | 20–28 | [Elo/form XGBoost script](../CSC410Pres/models/model_2/model_2.py) |
| AI-assisted serve data | 30–33 | [Serve experiment](../CSC410Pres/models/model_3/model_3.py), [supplied serve files](../CSC410Pres/data/serve_data) |
| AI-assisted injury reports | 34–37 | [Injury experiment](../CSC410Pres/models/model_4/model_4.py), [reports](../CSC410Pres/data/injury_reports) |
| AI-assisted playstyle categories | 38–41 | [Playstyle experiment](../CSC410Pres/models/model_5/model_5.py), [reports](../CSC410Pres/data/playstyle_reports) |
| Lessons and proposed next steps | 42–43 | Discussion in the deck; proposals are not implemented or validated outcomes |

The [combined model](../CSC410Pres/models/model_6/model_6.py) is also retained in the repository. The deck focuses on the separate feature experiments rather than giving that script a dedicated section.

## How to explain the work

The useful progression is from a simple baseline to richer historical features, then to experiments with harder-to-collect data. The AI component concerns obtaining and structuring additional player information; the prediction models themselves are logistic regression and XGBoost. The presentation's lessons about inconsistent outputs, sparse injury coverage and broad playstyle categories are part of the result, even when predictive accuracy did not improve.

The [SQLite preparation work](../CSC410Pres/processing/README.md) is a companion consolidation step. Existing model scripts continue to read their individual input files. Odds appear in the processed dataset and motivated the research, but the original model feature definitions do not use those odds directly. The repository does not demonstrate a profitable automated trading or betting system.

## Read the slide claims carefully

- The baseline slide combines a text summary with a chart from a different logistic comparison. The standalone baseline script and the logistic comparator inside the improved-model experiment are distinct runs on different match cohorts.
- Some slide tables do not match the reproduced outputs. The [results note](results.md) labels the measurements from saved predictions; the presentation remains unchanged as a record of what was presented.
- Dataset counts must distinguish physical matches from the mirrored player-order rows used for training. A mirrored row is not another independent match.
- Chronologically splitting train and test data does not, by itself, make full-period static lookups safe. The reproduced scores preserve those limitations.
- Feature importance describes how a fitted model uses its inputs. It does not prove that an AI-generated category is accurate or that the category causes a player to win. Accuracy also does not establish positive betting returns.

[TEAM NOTE: who built what]
