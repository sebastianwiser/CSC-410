from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from extensions.pytorch.model import TorchMLP
from extensions.point_in_time.features import BASE, SERVE, CAT
from portfolio.common import SEED


def make(kind='xgb', serve=False, params=None):
    numeric = BASE + (SERVE if serve else [])
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])
    if kind == 'mlp':
        clf = TorchMLP(seed=SEED)
    elif kind == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
    else:
        config = dict(n_estimators=200, max_depth=3, learning_rate=.05,
                      min_child_weight=15, subsample=.8, colsample_bytree=.8,
                      random_state=SEED, n_jobs=1, eval_metric='logloss')
        config.update(params or {})
        clf = XGBClassifier(**config)
    return Pipeline([('pre', pre), ('clf', clf)])
