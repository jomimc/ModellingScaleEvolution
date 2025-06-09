"""
This module, "cost_functions.py", contains code for calculating 
costs from scales, given a set of cost functions for models.

See the bottom of the code for details of the main functions to run,
in order to reproduce the work.

"""
import argparse
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import pickle
import time
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wasserstein_distance

from scales_io import *
import scales_utils as utils



N_PROC = 20     # Number of processors for parallel processing
USE_MP = True   # Set to False if not using parallel processing


TEMP_MIN = 90. 
TEMP_MAX = 300.
N_TRIALS = 200
TEMP_LOW_MARGIN = 0.50
TEMP_HI_MARGIN = 1.50

#################################################################
### Parameter ranges for models

### Window size (tolerance for deviation from exact intervals) for FIF and GP models
#W_ARR = np.arange(5, 30, 5)
W_ARR = np.arange(2, 42, 2)

### Exponent in GP model, where the model approaches
### the FIF model as n -> infinity
#N_ARR = np.array([1, 2, 3, 5, 10])
N_ARR = np.array([1])

### Exponent in Complexity model, which controls how cost of deviation from a template
### scales with the magnitude of the deviation
### This is also is used for altering the scale of the cost function
### (in a similar way to how Beta is used to control the magnitude of the cost)
#M_ARR = np.array([0.25, 0.5, 1, 2, 4])
#M_ARR = np.array([0.05, .10, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12])
M_ARR = np.array([0.05, .1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20])



### For Reference:
### FIF indices: cf_type, m, scale, w
### GP indices: n, cf_type, m, w, scale
### Complexity indices: m, cf_type, m OR w2, scale



### From a set of scales, generate either all scale intervals (all possible intervals),
### or all scale degrees. Octave equivalence is assumed for alg='circular'
def get_all_ints(ints, alg='circular'):
    if len(ints.shape) == 1:
        if alg == 'circular':
            # For each scale, representated as step intervals, calculate
            # all possible N*(N-1) intervals that can be made
            # (excluding the octave, since this simply amounts to a constant)

            return np.concatenate([np.cumsum(np.roll(ints, i)[:-1]) for i in range(ints.size)])

        elif alg == 'first':
            # For each scale, representated as step intervals, calculate
            # all possible N*(N+1)/2 intervals that can be made with respect to the first note

            return np.concatenate([np.cumsum(ints[i:]) for i in range(len(ints))])

    elif len(ints.shape) == 2:
        if alg == 'circular':
            # For each scale, representated as step intervals, calculate
            # all possible N*(N-1) intervals that can be made
            # (excluding the octave, since this simply amounts to a constant)

            return np.concatenate([np.cumsum(np.roll(ints, i, axis=1)[:,:-1], axis=1) for i in range(ints.shape[1])], axis=1)

        elif alg == 'first':
            # For each scale, representated as step intervals, calculate
            # all possible N*(N+1)/2 intervals that can be made with respect to the first note

            return np.concatenate([np.cumsum(ints[:,i:], axis=1) for i in range(ints.shape[1])], axis=1)




#################################################################
######################  Pre-processing

### Since these costs can take a while to calculate, it is more efficient
### to generate all possible values (which can be mapped to intervals) in advance.
### For scales that go beyond the octave, we do not count intervals beyond
### the octave (we actually use imax = 1250 cents, which is slightly higher than an octave)
def precompute_all(imax=1250, redo=0):

    path_models = PATH_COMP.joinpath("harmonicity_interference_model_scores.npy")
    path_names = PATH_COMP.joinpath("harmonicity_interference_model_names.txt")
    if path_models.exists() and not redo:
        return np.load(path_models), np.loadtxt(path_names, str)


    all_models = []
    model_names = []
    ints = np.arange(imax)

    # Gill - Purves model
    print("Calculating Gill-Purves scores")
    m, n = preprocess_all_gill_purves(imax)
    all_models.extend(m)
    model_names.extend(n)

    # FIF / OCT models
    print("Calculating FIF / OCT scores")
    m, n = preprocess_all_fifoct(ints)
    all_models.extend(m)
    model_names.extend(n)

    # Harrison-Pearce
    print("Calculating Harrison-Pearce scores")
    m, n = preprocess_all_harrison_pearce(imax)
    all_models.extend(m)
    model_names.extend(n)
    
    # Harmonic Overlap
    print("Calculating Harmonic Overlap scores")
    m, n = preprocess_all_harmonic_overlap(imax)
    all_models.extend(m)
    model_names.extend(n)

    # Interference 
    print("Calculating Interference scores")
    m, n = preprocess_all_interference(imax)
    all_models.extend(m)
    model_names.extend(n)


    all_models = normalize_models(np.array(all_models))
    model_names = np.array(model_names)

    np.save(path_models, all_models)
    np.savetxt(path_names, model_names, fmt='%s')

    return all_models, model_names


### Match parameters to models
def get_model_params():
    params = {'H_GP': [N_ARR, W_ARR],
              'H_WFIF': [W_ARR],
              'H_WOCT': [W_ARR],
              'H_WFO': [W_ARR],
              'H_GFIF': [W_ARR],
              'H_GOCT': [W_ARR],
              'H_GFO': [W_ARR]}

#   n_partial = list(range(2, 21)) + [50, 100, 200]
#   rho_arr = [0, 1, 2, 3, 5, 10, 20]
    n_partial = list(range(3, 41))
    rho_arr = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20] 
    params.update({'H_HP': [n_partial, rho_arr]})

    n_partial = [3, 10, 20]
    rho_arr = [0, 10, 20]
    params.update({'H_HO': [n_partial, rho_arr]})

    n_partial = list(range(2, 21)) + [50, 100, 200]
    rho_arr = [0, 1, 2, 3, 5, 10, 20]
    f0_arr = [50, 100, 200, 500, 1000]
    params.update({'I_HK': [n_partial, rho_arr, f0_arr]})
    params.update({'I_SE': [n_partial, rho_arr, f0_arr]})
    params.update({'I_BE': [n_partial, rho_arr, f0_arr]})

    names1 = ['rel', 'abs']
    params.update({'C_CI':[names1, M_ARR]})
    params.update({'C_NI':[W_ARR]})

    return params


### Names of parameters, matched to models
def get_param_names():
    params = {'H_GP': ["N", "W"],
              'H_WFIF': ["W"],
              'H_WOCT': ["W_ARR"],
              'H_WFO': ["W_ARR"],
              'H_GFIF': ["W_ARR"],
              'H_GOCT': ["W_ARR"],
              'H_GFO': ["W_ARR"],
              'H_HP': ["n", "rho"],
              'H_HO': ["n", "rho"],
              'I_HK': ["n", "rho", "f0"],
              'I_SE': ["n", "rho", "f0"],
              'I_BE': ["n", "rho", "f0"],
              'C_CI': ["RA", "M"],
              'C_NI': ["W"]}
    return params


### Normalize models so that they have values of similar magnitudes.
### This is useful because it means they will lead to similar
### optimal values of beta, which reduces the space of beta that we
### need to search through
def normalize_models(all_models):
    mstd = all_models.std(axis=1)
    mmin = all_models.min(axis=1)
    return (all_models - mmin.reshape(-1, 1)) / mstd.reshape(-1, 1)


def preprocess_all_gill_purves(imax):
    all_models, model_names = [], []
    har_gp_score = load_har_score(1, imax)
    for n in N_ARR:
        all_models.extend(list(har_gp_score**n))
        model_names.extend([f"H_GP_{n}_{w}" for w in W_ARR])
    return all_models, model_names


def preprocess_all_fifoct(ints):
    all_models, model_names = [], []

    # Basic FIF / OCT model
    fif_score = [(np.abs(ints - 702) <= w).astype(int) for w in W_ARR]
    all_models.extend(fif_score)
    model_names.extend([f"H_WFIF_{w}" for w in W_ARR])

    oct_score = [(np.abs(ints - 1200) <= w).astype(int) for w in W_ARR]
    all_models.extend(oct_score)
    model_names.extend([f"H_WOCT_{w}" for w in W_ARR])

    fif_oct_score = list(np.array(fif_score) + np.array(oct_score))
    all_models.extend(fif_oct_score)
    model_names.extend([f"H_WFO_{w}" for w in W_ARR])

    # Gaussian FIF / OCT model
    fif_score = [utils.gaussian(ints, 702, w) for w in W_ARR]
    all_models.extend(fif_score)
    model_names.extend([f"H_GFIF_{w}" for w in W_ARR])

    oct_score = [utils.gaussian(ints, 1200, w) for w in W_ARR]
    all_models.extend(oct_score)
    model_names.extend([f"H_GOCT_{w}" for w in W_ARR])

    fif_oct_score = list(np.array(fif_score) + np.array(oct_score))
    all_models.extend(fif_oct_score)
    model_names.extend([f"H_GFO_{w}" for w in W_ARR])

    return all_models, model_names


def hp_from_params(imax, n, rho, sigma):
    ints = np.arange(imax)
    partials, weights = utils.basic_timbre(ints, n=n, rho=rho)
    return calculate_harrison_pearce(partials, weights[0], sigma=sigma)


def preprocess_all_harrison_pearce(imax, mp=USE_MP, sigma=6.83):
    n_partial, rho_arr = get_model_params()['H_HP']
    sigma = 6.83
    if not mp:
        all_models, model_names = [], []
        for i, n in enumerate(n_partial):
            for j, rho in enumerate(rho_arr):
                print(i, j, '\t', n, rho)
                all_models.append(hp_from_params(imax, n, rho, sigma))
                model_names.append(f"H_HP_{n}_{rho}")
    else:
        inputs = [(imax, n, rho, sigma) for n in n_partial for rho in rho_arr]
        with Pool(N_PROC) as pool:
            res = np.array(list(pool.starmap(hp_from_params, inputs)))
        model_names = [f"H_HP_{n}_{rho}" for n in n_partial for rho in rho_arr]
        all_models = list(res.reshape(-1, imax))
    return all_models, model_names


def ho_from_params(imax, n, rho, sigma):
    ints = np.arange(imax)
    partials, weights = utils.basic_timbre(ints, n=n, rho=rho)
    return calculate_harmonic_overlap(partials, weights[0], sigma=sigma)


def preprocess_all_harmonic_overlap(imax, mp=USE_MP):
    n_partial, rho_arr = get_model_params()['H_HO']
    sigma = 6.83
    if not mp:
        all_models, model_names = [], []
        for i, n in enumerate(n_partial):
            for j, rho in enumerate(rho_arr):
                print(i, j, '\t', n, rho)
                all_models.append(ho_from_params(imax, n, rho, sigma))
                model_names.append(f"H_HO_{n}_{rho}")
    else:
        inputs = [(imax, n, rho, sigma) for n in n_partial for rho in rho_arr]
        with Pool(N_PROC) as pool:
            res = np.array(list(pool.starmap(ho_from_params, inputs)))
        model_names = [f"H_HO_{n}_{rho}" for n in n_partial for rho in rho_arr]
        all_models = list(res.reshape(-1, imax))
    return all_models, model_names


def interference_from_params(imax, n, rho, f0):
    return preprocess_interference_models(f0, n, rho, imax=imax)


def preprocess_all_interference(imax, mp=USE_MP):
    n_partial, rho_arr, f0_arr = get_model_params()['I_HK']
    all_models, model_names = [], []

    if not mp:
        diss_all = []
        for i, n in enumerate(n_partial):
            for j, rho in enumerate(rho_arr):
                for k, f0 in enumerate(f0_arr):
                    print(i, j, k, '\t', n, rho, f0)
                    diss_all.append(preprocess_interference_models(f0, n, rho, imax=imax))
    else:
        inputs = [(imax, n, rho, f0) for n in n_partial for rho in rho_arr for f0 in f0_arr]
        with Pool(N_PROC) as pool:
            diss_all = np.array(list(pool.starmap(interference_from_params, inputs)))

    # Changing the sign of dissonance, so that it matches harmonicity
    diss_all = - np.array(diss_all).reshape(len(n_partial), len(rho_arr), len(f0_arr), 4, imax)

    for m, name in zip([0, 2, 3], ['HK', 'SE', 'BE']):
        for i, n in enumerate(n_partial):
            for j, rho in enumerate(rho_arr):
                for k, f0 in enumerate(f0_arr):
                    all_models.append(diss_all[i,j,k,m])
                    model_names.append(f"I_{name}_{n}_{rho}_{f0}")

    return all_models, model_names



#################################################################
######################  HARMONICITY


#################################################################
### Window-based template matching (Gill & Purves style)


def harmonic_attractors_discrete(imax=1250):
    # Create a set of ratios, I / J
    # to get matrices for intervals in cents and harmonicity score
    I, J = np.meshgrid(*[range(200)]*2)
    cents_mat = 1200 * np.log2(I/J)
    score_mat = (I + J - 1) / (I * J)
    
    # Define the range of intervals, and window sizes to evaluate
    cents = np.arange(0, imax, 1)

    # Calculate the maximum score within each interval's window
    score = np.zeros((cents.size, W_ARR.size), float)
    for i, c in enumerate(cents):
        for j, w in enumerate(W_ARR):
            try:
                score[i,j] = np.max(score_mat[np.abs(cents_mat - c) <= w])
            except ValueError:
                score[i,j] = 0
    return cents, W_ARR, score.T


def load_har_score(redo=False, imax=1250):
    path = PATH_COMP.joinpath(f"GP_score.npy")
    if path.exists() and not redo:
        return np.load(path)

    return harmonic_attractors_discrete(imax)[2]


def calculate_harmonicity(ints, score, all_ints='', imax=1250):
    if isinstance(all_ints, str):
        all_ints = get_all_ints(ints, all_ints)

    # Scoring matrix does not go above 1250, so keep track of any intervals in this range
    idx = all_ints > imax
    all_ints[idx] = 0

    # Match ints to scores
    int_score = score[all_ints.astype(int)]

    # Assign no score to intervals above 1250
    int_score[idx] = np.nan

    # For each set of intervals,
    # calculate the mean of the harmonicity scores of each interval,
    # with equal weights for intervals
    return np.nanmean(int_score, axis=1).T


def calculate_int_freq_window(ints='', all_ints='', target=702):
    if isinstance(all_ints, str):
        all_ints = get_all_ints(ints, all_ints)

    # For each set of intervals, calculate the number of fifths
    out = []
    for w in W_ARR:
        int_score = (np.abs(all_ints - target) <= w).astype(float)
        out.append(np.mean(int_score, axis=1))
    return np.array(out)


def calculate_fif_window(ints='', all_ints=''):
    return calculate_int_freq_window(ints=ints, all_ints=all_ints, target=702)


def calculate_oct_window(ints='', all_ints=''):
    return calculate_int_freq_window(ints=ints, all_ints=all_ints, target=1200)


def calculate_octfif_window(ints='', all_ints=''):
    fif = calculate_int_freq_window(ints=ints, all_ints=all_ints, target=702)
    octave = calculate_int_freq_window(ints=ints, all_ints=all_ints, target=1200)
    return fif + octave


#################################################################
### Fifth / Octave count, continuous (Gaussian window)


def calculate_int_freq_gauss(ints='', all_ints='', target=702):
    if isinstance(all_ints, str):
        all_ints = get_all_ints(ints, all_ints)

    return np.array([np.mean(np.exp(-((all_ints - target) / w)**2), axis=1) for w in W_ARR])


def calculate_fif_gauss(ints='', all_ints=''):
    return calculate_int_freq_gauss(ints=ints, all_ints=all_ints, target=702)


def calculate_oct_gauss(ints='', all_ints=''):
    return calculate_int_freq_gauss(ints=ints, all_ints=all_ints, target=1200)


def calculate_octfif_gauss(ints='', all_ints=''):
    fif = calculate_int_freq_gauss(ints=ints, all_ints=all_ints, target=702)
    octave = calculate_int_freq_gauss(ints=ints, all_ints=all_ints, target=1200)
    return fif + octave



#################################################################
### Harrison-Pearce model


### Get the circular distance from 0 cents, between 0 and 1200 cents,
### where 0 and 1200 are equivalent.
### Then calculate the weight of each pitch class using a Gaussian distribution
def basic_pc_spectrum(sigma=6.83):
    X = np.arange(0, 1200)
    dist = np.min([X, 1200 - X], axis=0)
    return np.exp(-(dist / sigma)**2/2) / (sigma * (2 * np.pi)**0.5)


def get_pc_spectrum(partials, weights, f0=440, sigma=6.83):
    pc_set = np.round((np.log2(partials / f0) % 1) * 1200).astype(int)
    if isinstance(sigma, (int, float)):
        spec0 = basic_pc_spectrum(sigma=sigma)
    half = len(partials) // 2
    for i, pc in enumerate(pc_set):
        if isinstance(sigma, np.ndarray):
            spec0 = basic_pc_spectrum(sigma=sigma[i])
        if i == half:
            spec1 = dyad_spec.copy()
        if not i:
            dyad_spec = weights[i] * np.roll(spec0, pc)
        else:
            dyad_spec = dyad_spec + weights[i] * np.roll(spec0, pc)
    return spec0, spec1, dyad_spec


def get_vpc_spectrum(partials, weights, sigma=6.83):
    spec1, dyad_spec = get_pc_spectrum(partials, weights, sigma=sigma)[1:]
    conv =  utils.conv_circ(dyad_spec, spec1)
    # Not sure why, but we need to circularly permute the output
    conv_norm = conv / np.linalg.norm(spec1) / np.linalg.norm(dyad_spec)
    return conv_norm

    
def calculate_harrison_pearce(partials, weights, sigma=6.83):
    return np.array([utils.get_kl_div_uniform(get_vpc_spectrum(p, weights, sigma=sigma)) for p in partials])


def calculate_harmonic_overlap(partials, weights, sigma=6.83):
    return np.array([utils.harmonic_overlap(p, weights) for p in partials])



#################################################################
######################  INTERFERENCE


def diss_mod_fn(xp, D, q=1.632):
    return xp * D - q * (1 - xp) * (1 + np.sin(2 * np.pi * xp - np.pi / 2))


def slow_beat_modification(bw, diss, p=0.096):
    if isinstance(diss, (int, float)):
        if diss > p:
            return diss
        else:
            return diss_mod_fn(bw / p, diss)

    elif isinstance(diss, np.ndarray):
        idx = diss < p
        if np.sum(idx):
            diss[idx] = diss_mod_fn(bw[idx] / p, diss[idx])
        return diss

    else:
        raise TypeError(f"Wrong type passed to slow_beat_modification: {type(d)}")


def critical_bandwidth_hk(f1, f0):
    return np.abs(f1 - f0) / (1.72 * ((f1 + f0) / 2)**0.65)
 

def dissonance_hutchinson_knopoff(f1, f0):
    bw = critical_bandwidth_hk(f1, f0)
    bw[bw > 1.2] = 0
    return (4 * bw * np.exp(1 - 4 * bw))**2


def dissonance_hutchinson_knopoff_revised(f1, f0, a=4, b=2):
    bw = critical_bandwidth_hk(f1, f0)
    bw[bw > 1.2] = 0
    diss = (a * bw * np.exp(1 - a * bw))**b
    return slow_beat_modification(bw, diss)
    

def dissonance_sethares(f1, f0, b1=3.5, b2=5.75, dstar=0.24, s1=0.021, s2=19):
    x = np.abs(f1 - f0)
    fmin = np.min([f1, f0], axis=0)
    s = dstar / (s1 * fmin + s2)
    return (np.exp(-b1 * s * x) - np.exp(-b2 * s * x))


def dissonance_berezovsky(f1, f0):
    interval = np.abs(np.log2(f1 / f0))
    if isinstance(f1, (int, float)):
        wc = 0.67 * min(f1, f0)**(-0.68)
    elif isinstance(f1, np.ndarray):
        wc = 0.67 * np.min([f1, f0], axis=0)**(-0.68)
    return np.exp(-(np.log(interval / wc)**2)) / wc


def preprocess_interference_models(f1=440, n_partial=10, rho=3, r_exp=1.359, imax=1250):
    ints = np.arange(0, imax).astype(float)

    model_names = ['HK', 'HK_rev', 'Seth']

    # Get frequencies of partials
    partials, weights = utils.basic_timbre(ints, f1=f1, n=n_partial, rho=rho)
    # Weights are independent of the interval
    weights = weights[0]

    # Get indices for pairs of partials
    i0, j0 = np.triu_indices(partials.shape[1], 1)

    # For each pair, calculate dissonance between pairs
    diss_hk = np.array([dissonance_hutchinson_knopoff(partials[:,i], partials[:,j]) for i, j, in zip(i0, j0)])
    diss_hk_rev = np.array([dissonance_hutchinson_knopoff_revised(partials[:,i], partials[:,j]) for i, j, in zip(i0, j0)])
    diss_seth = np.array([dissonance_sethares(partials[:,i], partials[:,j]) for i, j, in zip(i0, j0)])
    diss_bere = np.array([dissonance_berezovsky(partials[:,i], partials[:,j]) for i, j, in zip(i0, j0)])

    # Calculate weights of pairs
    w_hk = np.outer(weights, weights)[i0, j0] / np.sum(weights**2)
    w_hk_rev = np.outer(weights, weights)[i0, j0]**(r_exp / 2.) / np.sum(weights**r_exp)
    w_seth = np.min([weights[i0], weights[j0]], axis=0)
    w_bere = weights[i0]**0.606

    # Calculate weights of pairs
    diss_hk = np.average(diss_hk, weights=w_hk, axis=0)
    diss_hk_rev = np.average(diss_hk_rev, weights=w_hk_rev, axis=0)
    diss_seth = np.average(diss_seth, weights=w_seth, axis=0)
    diss_bere = np.average(diss_bere, weights=w_bere, axis=0)
    return np.array([diss_hk, diss_hk_rev, diss_seth, diss_bere])
    

#################################################################
######  Scale Complexity


def template_function_single(ints, M=1, RA='rel'):
    mi = max(TEMP_MIN, TEMP_LOW_MARGIN * min(ints))
    ma = TEMP_MAX
    baseArr = np.linspace(mi, ma, N_TRIALS)
    ints = ints.reshape(1, -1)
    baseArr = baseArr.reshape(-1, 1)
    ratio = ints / baseArr
    if RA == 'rel':
        err = np.abs(np.round(ratio) - ratio)
    else:
        err = np.abs(np.round(ratio) * baseArr - ints) / 100
    return np.mean(err**M, axis=1).min()


### This function checks how well a set of intervals fits
### onto a simple integer series
def template_function_many(ints, m_arr=M_ARR):
    N = len(ints)
    Imin = ints.min(axis=1)
    mi = np.max([np.ones(N)*TEMP_MIN, TEMP_LOW_MARGIN * Imin], axis=0).reshape(-1,1)
    ma = np.ones((N,1)) * TEMP_MAX
    baseArr = np.array([np.linspace(0,1,N_TRIALS)]*N) * (ma - mi) + mi

    ints = ints.reshape(ints.shape + (1,))
    baseArr = baseArr.reshape(-1, 1, N_TRIALS)
    ratio = ints / baseArr

    # Either relative or absolute deviation from template is used
    err1 = np.abs(np.round(ratio) - ratio)
    # Dividing by 100 to get error in semitones (to avoid large numbers)
    err2 = np.abs(np.round(ratio) * baseArr - ints) / 100
    return np.array([[np.mean(e**m, axis=1).min(axis=1) for m in m_arr] for e in [err1, err2]])


### This function calculates the number of unique intervals
### in a set of intervals by clustering
def number_unique_intervals(ints, W=None):
    if len(ints) <= 1:
        return np.ones(W_ARR.size) * len(ints)

    li = linkage(ints.reshape(-1, 1), method='ward', metric='euclidean')
    max_var = []
    # Add intervals to clusters one at a time, and record the maximum variance
    for threshold in list(li[:,2][::-1]) + [0]:
        clust = fcluster(li, threshold, criterion='distance')
        max_var.append(max([np.std(ints[clust==c]) for c in np.unique(clust)]))
    max_var = np.array(max_var)

    # Get the number of clusters as a function of maximum standard deviation
    if isinstance(W, type(None)):
        return np.array([np.where(max_var < w)[0][0] + 1 for w in W_ARR])
    else:
        return np.where(max_var < W)[0][0] + 1
        


def number_unique_intervals_many(ints, W=None):
    return np.array([number_unique_intervals(i, W) for i in ints]).T


def info_model_names():
    names1 = ['rel', 'abs']
    cf_names = []
    for N in 'IA':
        cf_names.extend([f"C_CI_{N}_{n1}_{m}" for n1 in names1 for m in M_ARR])
        cf_names.extend([f"C_NI_{N}_{w}" for w in W_ARR])
    return np.array(cf_names)


def calculate_all_info_cf(ints, ai_alg='first'):
    all_ints = get_all_ints(ints, ai_alg)
    all_cf = []
    cf_names = []
    names1 = ['rel', 'abs']
    for I, N in zip([ints, all_ints], ['I', 'A']):
        all_cf.extend(template_function_many(I).reshape(-1, len(I)))
        cf_names.extend([f"C_CI_{N}_{n1}_{m}" for n1 in names1 for m in M_ARR])

        all_cf.extend(number_unique_intervals_many(I).reshape(-1, len(I)))
        cf_names.extend([f"C_NI_{N}_{w}" for w in W_ARR])
    return np.array(all_cf), np.array(cf_names)




if __name__ == "__main__":
    precompute_all(1250, True)


