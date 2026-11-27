"""Invoke original loaders, feature builders and estimators, not their test-reusing mains."""
import numpy as np
import pandas as pd
from portfolio.common import legacy_module, DATA


def processed_identity(df):
    # A single identity per source match, carried through both player perspectives.
    return pd.DataFrame({'match_id': ['processed:' + str(i) for i in range(len(df))],
                         'date': df.Date.to_numpy(), 'player1': df.Player_1.to_numpy(),
                         'player2': df.Player_2.to_numpy(), 'surface': df.Surface.to_numpy()})


def build(number):
    m = legacy_module(number)
    if number == 1:
        raw = m.load_raw(str(DATA / 'initial'))
        data = m.build_dataset(raw)
        ids = pd.DataFrame({'match_id': ['raw:' + str(i) for i in range(len(raw))],
                            'date': raw.match_date, 'player1': raw.winner_name,
                            'player2': raw.loser_name, 'surface': raw.surface.str.lower()})
        # Original Model 1 interleaves winner/loser rows.
        meta = ids.loc[ids.index.repeat(2)].reset_index(drop=True)
        meta['perspective'] = np.tile([0, 1], len(raw))
    else:
        df = m.load_processed(m.PROC_DIR)
        ids = processed_identity(df)
        matches = m.load_initial_matches(m.INIT_DIR)
        attrs = m.build_player_attrs(matches)
        surfaces = m.build_surface_lookup(m.load_surface_results(m.INIT_DIR))
        df = m.compute_dynamic_features(df)
        args = [df, attrs, surfaces]
        if number in (3, 6):
            args.append(m.build_serve_lookup(m.load_serve_data(m.SERVE_DIR)))
        if number == 4:
            args.append(m.load_injury_data(m.INJURY_DIR))
        if number in (5, 6):
            style = m.load_playstyle_data(m.PLAYSTYLE_DIR)
            args.append(style[0] if number == 5 else style)
        if number == 6:
            args.append(m.build_player_clusters(matches)[0])
        data = m.build_dataset(*args)
        meta = pd.concat([ids, ids], ignore_index=True)
        meta['perspective'] = np.repeat([0, 1], len(ids))
    cat = getattr(m, 'CAT_COLS_M5', None) if number == 5 else m.CAT_COLS
    num = m.NUMERIC_COLS
    data = pd.concat([data.reset_index(drop=True), meta.drop(columns=[c for c in meta if c in data.columns])], axis=1)
    factory = getattr(m, {4: 'make_m4_pipeline', 5: 'make_m5_pipeline',
                          6: 'make_m6_pipeline'}.get(number, 'make_pipeline'))
    return m, data, num + cat, factory
