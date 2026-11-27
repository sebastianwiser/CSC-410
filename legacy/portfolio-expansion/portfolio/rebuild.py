"""Recreate the selected model artifact using training data only; never score test data."""
import json
import joblib
from portfolio.common import RESULTS, write_json, fingerprints
from portfolio.models import make
from extensions.point_in_time.features import load_raw, build


def main():
    config = json.loads((RESULTS/'validation.json').read_text())
    name = config['selected']
    data = build(load_raw())
    train = data[data.year.between(2014, 2020)]
    factory = {'safe_logistic': lambda: make('logistic'), 'safe_xgb': lambda: make(),
               'safe_xgb_tuned': lambda: make(params=config['params']),
               'safe_mlp': lambda: make('mlp'), 'point_in_time_serve': lambda: make(serve=True)}[name]
    model = factory()
    features = model.named_steps['pre'].transformers[0][2] + model.named_steps['pre'].transformers[1][2]
    model.fit(train[features], train.label)
    path = RESULTS/'artifacts'/f'{name}.joblib'
    path.parent.mkdir(exist_ok=True)
    joblib.dump(dict(model=model, features=features, name=name), path)
    print(f'Rebuilt {name} from training rows only: {len(train)}')


if __name__ == '__main__':
    main()
