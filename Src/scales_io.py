"""
This module contains functions for reading files.
"""
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scales_config import *
import scales_utils as utils


def load_all_data(redo=False):
    path_df = PATH_DATA.joinpath("Scales", "all_scales_data.pkl")
    if not path_df.exists() or redo:
        names = ['damusc', 'steven', 'garland']
        df_list = [pd.read_pickle(PATH_DATA.joinpath("Scales", f"dataset_{n}.pkl")) for n in names]
        df = pd.concat(df_list, ignore_index=True)
        df['ScaleType'] = df.Instrument.apply(utils.identify_scale_type)
        df['n_notes'] = df.Scale.apply(len)
        df['n_ints'] = df.n_notes - 1
        df['scale_range'] = df.Scale.apply(utils.scale_range)
        df['int_min'] = df.Ints.apply(utils.int_min)
        df['int_max'] = df.Ints.apply(utils.int_max)
        df['int_mean'] = df.Ints.apply(np.mean)
        df['int_std'] = df.Ints.apply(np.std)

        df = utils.rename_regions(df)
        df = utils.rename_countries(df)

        # Remove single-note scales
        df = df.loc[df.n_ints>0]

        df.to_pickle(path_df)

    else:
        df = pd.read_pickle(path_df)

    return df


def load_int_dist():
    int_set, prob = np.load(PATH_DATA.joinpath("Scales", "step_sizes_kde.npy")).T
    prob = prob / prob.sum()
    return int_set, prob


def load_int_dist_n(df, n, xmax=1201, redo=False):
    path = PATH_COMP.joinpath("IntDist", f"{n}.npy")
    if path.exists() and not redo:
        return np.load(path)
    X = np.arange(xmax)
    ints = [x for y in df.loc[df.n_ints==n, 'Ints'] for x in y]
    kde = gaussian_kde(ints)(X)
    np.save(path, kde)
    return kde


