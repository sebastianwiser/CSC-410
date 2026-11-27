# CSC 410: ATP Tennis Match Prediction

A school project exploring how historical tennis results and AI-assisted data collection can support match prediction. The work was developed around a class presentation: start with a logistic baseline, improve it with tennis-specific history, then test whether extra serve, injury and playstyle information helps.

**[View the class presentation](docs/presentation.pdf)** · [Follow the presentation through the code](docs/presentation-guide.md) · [Read the reproduced results](docs/results.md)

[TEAM NOTE: who built what]

## The project story

- **Establish a baseline.** [Logistic regression](CSC410Pres/models/model_1/model_1.py) uses rankings, age, head-to-head history and match context.
- **Add tennis-specific history.** [XGBoost with Elo and form](CSC410Pres/models/model_2/model_2.py) adds overall/surface Elo, recent results, rest, experience and player/surface attributes.
- **Explore AI-assisted data collection.** Separate experiments add [serve statistics](CSC410Pres/models/model_3/model_3.py), [injury reports](CSC410Pres/models/model_4/model_4.py) and [playstyle categories](CSC410Pres/models/model_5/model_5.py). The presentation discusses both the modeling ideas and the difficulty of getting consistent, useful AI-generated data.
- **Compare what helped.** The repository also retains a [combined model](CSC410Pres/models/model_6/model_6.py). Its role is documented alongside the individual experiments; it is not a separate section of the supplied presentation.

The main learning was about data preparation, chronological features, model comparison and the limits of AI-collected data. Betting markets motivated the original idea; a profitable betting strategy was not established.

## What the reproduced results show

The ranking/Elo/form model improved on its same-data logistic comparator. Injury and playstyle additions did not improve reproduced accuracy. Serve features and the combined model showed small changes, but their paired accuracy intervals include no improvement. These results support a cautious class-project conclusion rather than a claim that every extra data source helped.

The [results note](docs/results.md) contains measured scores, unique match counts and the important differences from the slides. In particular, the standalone logistic script and the logistic comparator inside the improved-model experiment use different datasets. Slide figures are retained as historical presentation material; they are not substituted for reproduced measurements.

## Repository guide

| Location | What it contains |
|---|---|
| [Presentation](docs/presentation.pdf) | The supplied classroom deck, unchanged |
| [CSC410Pres/models](CSC410Pres/models) | Original model scripts, plots and saved artifacts |
| [CSC410Pres/data](CSC410Pres/data) | Checked-in match data and supplemental reports |
| [CSC410Pres/processing](CSC410Pres/processing/README.md) | SQLite consolidation script and the data-formatting diagram |
| [docs](docs) | Presentation walkthrough and concise results notes |
| [legacy](legacy/README.md) | Earlier loose experiments and the later portfolio expansion |

## Set up and explore

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_project.py
```

The check uses saved predictions to verify the reported scores and counts. It also checks original source hashes and documentation links. It does not retrain models, make new holdout predictions or call an AI service.

To run an original class model yourself:

```bash
python CSC410Pres/models/model_2/model_2.py
```

Each model folder has its corresponding script. The original scripts retain their historical validation routines and can overwrite their own plots and saved model artifacts. Their full runs are different from the lightweight saved-evidence check above. The [SQLite processing guide](CSC410Pres/processing/README.md) is optional; the model scripts still read their original input files.

## Limitations

- Serve, injury and playstyle reports may be LLM-sourced, including `gemini_*` / `openai_*` files. Their completeness, accuracy and collection dates are not fully verified.
- Original static player/surface lookups and combined-model clusters use full-period information. Same-date history and non-temporal calibration also limit claims of leakage-free performance. Original model logic is preserved.
- Different data sources cover different matches. Match-bootstrap intervals assume independence despite repeated players/events, and the reported comparisons do not establish betting profitability or causal explanations of player performance.

The later API, Docker, PyTorch, tuning, MLflow and SHAP additions are preserved in the [portfolio archive](legacy/portfolio-expansion/README.md). They are outside the classroom presentation and are not required for the main setup.
