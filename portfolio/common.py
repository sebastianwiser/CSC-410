"""Shared experiment configuration. No legacy script is modified."""
from pathlib import Path
import hashlib
import json
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'CSC410Pres/data'
RESULTS = ROOT / 'results'
SEED = 42
TRAIN_START, TRAIN_END, TEST_START, TEST_END = 2014, 2020, 2021, 2024
VALIDATION_YEARS = (2018, 2019, 2020)


def legacy_module(number):
    name = f'legacy_model_{number}'
    path = ROOT / f'CSC410Pres/models/model_{number}/model_{number}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def fingerprints():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DATA.rglob('*')) if p.is_file()}
