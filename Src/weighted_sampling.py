"""
This module, "weighted_sampling.py", contains code for sample reweighting.

Since there are several types of scales, separated into geographic regions,
not all of the categories and regions are equally represented in the data.

This module contains code for reweighting for scale model comparison,
and for step size, scale degree, and scale interval distributions.

"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde



def get_sample_key(df, nmax, xcat='Region'):
    regions = df[xcat].unique()
    return {k: min(v, nmax) for k, v in df[xcat].value_counts().items()}


def get_sample_weights(df, nmax, xcat='Region'):
    regions = df[xcat].unique()
    reg_weight = {k: min(v, nmax) / v for k, v in df[xcat].value_counts().items()}
    return df[xcat].map(reg_weight)


#################################################################
### Step interval distributions

def get_weighted_step_dist_inst(df, nmax=20):
    df = df.loc[df.ScaleType=="Instrumental"]
    df['weight'] = get_sample_weights(df, nmax)

    step = np.array([x for y in df.loc[(df.n_notes>1), "Ints"] for x in y])
    weight = np.array([x for y, z in zip(*df.loc[(df.n_notes>1), ["Ints", "weight"]].values.T) for x in [z]*len(y)])

    return step, weight / weight.sum()

    
def get_weighted_step_dist_vocal(df, nmax=40):
    df = df.loc[df.ScaleType=="Vocal"]
    df['weight'] = get_sample_weights(df, nmax)

    step = np.array([x for y in df.loc[(df.n_notes>1), "Ints"] for x in y])
    weight = np.array([x for y, z in zip(*df.loc[(df.n_notes>1), ["Ints", "weight"]].values.T) for x in [z]*len(y)])

    return step, weight / weight.sum()


def get_weighted_kde(X, W, bw=0.2):
    xgrid = np.arange(0, 801)
    return xgrid, gaussian_kde(X, bw_method=bw, weights=W)(xgrid)


def get_step_distributions_by_region(df):
    n_max = np.arange(10, 160, 5)
    inst_kde = {n: get_weighted_kde(*get_weighted_step_dist_inst(df, n)) for n in n_max}
        
    n_max = np.arange(10, 80, 5)
    vocal_kde = {n:get_weighted_kde(*get_weighted_step_dist_vocal(df, n)) for n in n_max}
    
    return inst_kde, vocal_kde


#################################################################
### Scale model weights


def get_weighted_scales(df, nmax=20, scale_type='Vocal'):
    weights = np.zeros(len(df), float)
    if scale_type != '':
        idx = df.ScaleType == scale_type
    else:
        idx = np.ones(len(df), bool)
    df = df.loc[idx]
    weights[idx] = get_sample_weights(df, nmax)
    return weights

    
def get_weights_by_region(df):
    n_ints = np.arange(2, 26)
    n_max = np.arange(0, 105, 5)
    scale_types = ['', 'Vocal', 'Instrumental', 'Theory']
    weights = {ni: [] for ni in n_ints}
    for ni in n_ints:
        for st in scale_types:
            weights[ni].append([get_weighted_scales(df.loc[df.n_ints==ni], nm, st) for nm in n_max])
        weights[ni] = np.array(weights[ni])

    return fix_weights(weights)


def get_weights_by_region_overall(df):
    n_ints = np.arange(2, 10)
    n_max = np.arange(0, 105, 5)
    scale_types = ['', 'Vocal', 'Instrumental', 'Theory']
    weights = []
    for st in scale_types:
        weights.append([get_weighted_scales(df, nm, st) for nm in n_max])
    weights = np.array(weights)

    return fix_weights(weights)


def get_weights_within_regions(df):
    n_ints = np.arange(2, 26)
    scale_types = ['', 'Vocal', 'Instrumental', 'Theory']
    regions = np.array(sorted(df.Region.unique()))
    weights = {ni: [] for ni in n_ints}
    for ni in n_ints:
        df0 = df.loc[df.n_ints==ni]
        for st in scale_types:
            if len(st):
                weights[ni].append([np.array((df0.ScaleType==st)&(df0.Region==r), float) for r in regions])
            else:
                weights[ni].append([np.array((df0.Region==r), float) for r in regions])
        weights[ni] = np.array(weights[ni])
    return fix_weights(weights)


### Set weights to zero on problematic scales
def fix_weights(weights):
    if isinstance(weights, dict):
        # Ignore one weird scale taken from a set of horns;
        # this one completely messes up the calculation due
        # to having an octave as a step interval
        weights[5][:,:,5] = 0

        # Ignore M0413 because it has unison intervals, and this
        # leads to extremely high harmonicity scores, messing up the
        # overall analysis
        weights[9][:,:,11] = 0

        # This one is a duplicate
        weights[8][:,:,7] = 0

    else:
        idx = [274, 1097, 1161]
        weights[:,:,idx] = 0

    return weights


### Smoothed scale degree probability density
def get_weighted_kde(df, st='Vocal', n=2, nmax=25, xmin=50, xmax=1250, dx=1):
    df = df.loc[(df.ScaleType==st)&(df.n_ints==n)]
    scales = np.array([np.cumsum(x) for x in df['Ints']])
    W = np.array([get_sample_weights(df, nmax)] * n).T
    X = np.arange(xmin, xmax+dx, dx)
    return X, gaussian_kde(scales.ravel(), bw_method=0.05, weights=W.ravel())(X)



