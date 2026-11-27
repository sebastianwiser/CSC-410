import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from portfolio.common import SEED


def metrics(y, p):
    return dict(accuracy=float(accuracy_score(y, p >= .5)),
                roc_auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)))


def bootstrap_mean(values, draws=2000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(values, dtype=float)
    out = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return np.quantile(out, [.025, .975]).tolist()


def paired(a, b):
    joined = a.merge(b, on='match_id', suffixes=('_a', '_b'), validate='one_to_one')
    if joined.empty:
        return {'n': 0, 'reason': 'No confidently aligned source matches'}
    if not (joined.label_a == joined.label_b).all():
        raise ValueError('Paired labels differ; orientation alignment failed')
    ca = (joined.probability_a >= .5).to_numpy() == joined.label_a.to_numpy()
    cb = (joined.probability_b >= .5).to_numpy() == joined.label_b.to_numpy()
    delta = ca.astype(float)-cb.astype(float)
    wins, losses = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(n=len(joined), accuracy_difference=float(delta.mean()),
                ci95=bootstrap_mean(delta), discordant_wins=wins, discordant_losses=losses,
                mcnemar_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.)
