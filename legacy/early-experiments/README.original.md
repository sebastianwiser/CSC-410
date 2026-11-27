# CSC-410
Tennis analytics project

## Simple baseline model
This repo includes a very small baseline model that predicts a match outcome
using only pre-match features (rank, age, height, surface, etc.).

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train and evaluate
```bash
python simple_model.py
```

By default it trains on 2014-2022 and tests on 2023-2024, then saves
`model.joblib` in the project root and charts in `plots/`. It also prints
a simple baseline accuracy that always picks the better (lower) rank.
You can change the split:
```bash
python simple_model.py --train-year-start 2016 --train-year-end 2021 --test-year-start 2022 --test-year-end 2024
```
