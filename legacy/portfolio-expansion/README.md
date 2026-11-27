# Archived portfolio expansion

These additions were created after the original class project was reviewed. They
are preserved for inspection, but the active repository now tells the story of the
class presentation and the models in `CSC410Pres/`.

`README.portfolio.md` and `MODEL_CARD.md` preserve the earlier portfolio framing.
`results/` retains measured scores, per-match predictions, statistical comparisons,
AI-data audits and run summaries. Its dates, run IDs and absolute local paths are
historical records. The older audit and SHAP scopes are retained in `legacy/`.

The optional packages and large dependency lock are kept here rather than in the
class project's setup. API, Docker and training-selection tools are archived too.
`ci/portfolio-tests.yml` is an inactive copy of the former workflow. The only active
workflow now performs lightweight class-project checks.

A relative `CSC410Pres` link preserves the archived tools' data and model paths.
If you inspect or run an archived tool, start from this folder and use its archived
dependency file. The tools write their reports here rather than overwriting the
class project's root README. Commands in the historical README describe the old
layout; old MLflow stores or external data sources may still need reconstruction.
Do not rerun final holdout evaluations to choose among models.
