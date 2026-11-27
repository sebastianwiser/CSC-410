"""One outcome-independent orientation and one prediction per physical match."""
import hashlib
import numpy as np
import pandas as pd


def canonical(data, probabilities):
    d = data[['match_id', 'date', 'player1', 'player2', 'surface', 'perspective', 'label']].copy()
    d['probability'] = probabilities
    a = d[d.perspective == 0].set_index('match_id')
    b = d[d.perspective == 1].set_index('match_id').loc[a.index]
    # Symmetrize both orientations before scoring; each source match contributes once.
    a['probability'] = .5*(a.probability + 1-b.probability)
    flip = np.array([int(hashlib.sha256(i.encode()).hexdigest()[-1], 16) % 2 for i in a.index], dtype=bool)
    a.loc[flip, 'probability'] = 1-a.loc[flip, 'probability']
    a.loc[flip, 'label'] = 1-a.loc[flip, 'label']
    a.loc[flip, ['player1', 'player2']] = a.loc[flip, ['player2', 'player1']].to_numpy()
    a['perspective'] = flip.astype(int)
    return a.reset_index()
