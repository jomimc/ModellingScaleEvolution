"""
This module, "evaluate_models.py", solves the general problem of given a set of models,
each with cost functions that depend on scales, which model best reproduces
a population of empirical scales.

This should be run in steps.
1. Get normalization constants for all models.
    This is computationally expensive, and constants need to be checked
    for convergence. Precomputed constants are provided on Zenodo.
2. Calculate likelihods for all empirical scales
3. Convert likelihoods into probabilities using the normalization constants.
4. Calculate model probabilities for each value of n_ints (number of steps in a scale)
5. Finally, the function "best_model_across_n" can be used to integrate
    the model probabilities across values of n_ints, while weighting scales
    to avoid overrepresentation of specific regions.
    This function is called in "paper_figs.py" and "si_figs.py" to create
    main and supplementary figures.

See the bottom of the code for details of the main functions to run
for preprocessing of steps 1-4.

"""
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import cost_functions as CF
import interval_constraints as IC
from scales_config import *
import scales_io
import scales_utils as utils
import weighted_sampling as WS


N_BETA = 101
MIN_BETA = 10.**-2
MAX_BETA = 10.**4
BETA_ARR = (10.**np.linspace(np.log10(MIN_BETA), np.log10(MAX_BETA), N_BETA))

N_INTS = np.arange(2, 26)

SCORE, NAMES = CF.precompute_all()
SCORE[:,0] = np.nan


#####################################################
### Generate scales

### Generate a scale by sampling steps from an interval distribution.
### The resulting step distribution should be approximate the input distribution.
def generate_scales_sampling(int_set, prob, n_ints, n_scales, octave=False):
    ints = np.random.choice(int_set, p=prob, size=n_ints * n_scales)
    ints = ints.reshape(n_scales, n_ints)
    if octave:
        ints = ints / (np.sum(ints, axis=1) / 1200).reshape(-1, 1)
    return np.round(ints).astype(int)


#####################################################
### Calculate likelihoods from ints

### Given an array of costs, return an array of likelihoods with one extra dimension (N_BETA)
def get_likelihood_from_cost(cost, bmin=MIN_BETA, bmax=MAX_BETA, nbeta=N_BETA):
    beta_arr = (10.**np.linspace(np.log10(bmin), np.log10(bmax), nbeta))
    return np.exp(-np.outer(beta_arr, cost.ravel())).reshape(beta_arr.shape + cost.shape)


### HI scores are positive, but they need to be minimized, so we take
### the negative or inverse.
### When taking the inverse, we add a constant to avoid division by zero.
def transform_HI_functions(cost):
    return np.array([-cost, (1 / (1 + cost))])


### No transformation necessary
def transform_info_functions(cost):
    return cost


def get_cost_from_ints(ints, imax=1250, ai_alg='first'):
    n_scales = len(ints)
    all_ints = CF.get_all_ints(ints, ai_alg)

    # Create versions of 'ints' with zeros for values
    # over 'imax', which will end up with 'nan' scores
    ints0 = ints.copy()
    all_ints0 = CF.get_all_ints(ints0, ai_alg)
    ints0[ints >= imax] = 0
    all_ints0[all_ints >= imax] = 0

    ### There are two ways to transform the positive (by definition) HI scores into a cost,
    ### while the info functions are already in the form of a cost.
    ### The resulting costs should have shapes:
    ###      (nbeta,  2 * num_HI_models, n_scales)
    ###      (nbeta,  2 * num_HI_models, n_scales)
    ###      (nbeta,  num_info_models * 2, n_scales)
    ###
    cost = np.concatenate([transform_HI_functions(np.nanmean(SCORE[:, ints0], axis=-1)).reshape(-1, n_scales),
                           transform_HI_functions(np.nanmean(SCORE[:, all_ints0], axis=-1)).reshape(-1, n_scales),
                           transform_info_functions(CF.calculate_all_info_cf(ints)[0]).reshape(-1, n_scales)], axis=0)
    return cost



#####################################################
### Estimate normalization factors (Z in the paper)

### Generate scales by sampling "n_ints" steps from a probability distribution ("int_set", "prob"),
### and calculate likelihoods (per model).
### Generate "n_scales" scales at a time (this is limited by memory, since "likelihood" is a large array),
### and repeat the process "nrep" times to get an average likelihood (per model).
def estimate_model_norm(int_set, prob, n_ints, n_scales=1000, nrep=1000, imax=1250, octave=False):
    ai_alg = 'circular' if octave else 'first'
    for i in range(nrep):
        ints = generate_scales_sampling(int_set, prob, n_ints, n_scales, octave=octave)
        likelihood = get_likelihood_from_cost(get_cost_from_ints(ints, imax, ai_alg))

        if i == 0:
            mean = likelihood.mean(axis=-1)
        else:
            mean = mean + likelihood.mean(axis=-1)
    return mean / nrep


### Estimate a set of average likelihoods
def estimate_average_run_one(i, n, octave=False, n_scales=10, nrep=1000, imax=1250, redo=1):
    if octave:
        path = PATH_COMP.joinpath("CostFn", "Normalization", f"octave_{n}_{n_scales}_{nrep}_{i:05d}.npy")
    else:
        path = PATH_COMP.joinpath("CostFn", "Normalization", f"{imax}_{n}_{n_scales}_{nrep}_{i:05d}.npy")
    if path.exists() and not redo:
        return
    print(path)

    int_set, prob = scales_io.load_int_dist()
    mean = estimate_model_norm(int_set, prob, n, n_scales, nrep, imax, octave=octave)
    np.save(path, mean)


### Run to get many sets of average likelihoods. These can be used to get a very accuate
### approximation of the true average, and to measure convergence.
def estimate_average_run_all():
    nrep = np.arange(200)
    octave = [False, True]
    chunk = N_PROC // (N_INTS.size * nrep.size) + 1
    with Pool(N_PROC) as pool:
        pool.starmap(estimate_average_run_one, product(nrep, N_INTS, octave), chunk)



#####################################################
### Calculate likelihood for real scales


def calculate_likelihood_real_scales(n, imax=1250, redo=False, octave=False):
    df = scales_io.load_all_data()
    ai_alg = 'circular' if octave else 'first'
    txt = 'octave' if octave else str(imax)
    path_cost = PATH_COMP.joinpath("CostFn", "EmpiricalCost", f"{txt}_{n}.npy")
    path_like = PATH_COMP.joinpath("CostFn", "EmpiricalLike", f"{txt}_{n}.npy")

    if path_like.exists() and not redo:
        return

    if n not in df.n_ints.values:
        return

    ints = np.array([x for x in df.loc[df.n_ints==n, 'Ints']]).astype(int)
    cost = get_cost_from_ints(ints, imax, ai_alg)
    np.save(path_cost, cost)

    likelihood = get_likelihood_from_cost(cost)
    print(path_like)
    np.save(path_like, likelihood)


### Calculate the costs and likelihoods (per model) for all scales
def run_all_likelihood_real_scales(imax=1250, redo=False):
    n_ints = np.arange(2, 26)
    octave_list = [False, True]
    with Pool(N_PROC) as pool:
        inputs = [(n, imax, redo, o) for n, o in product(n_ints, octave_list)]
        pool.starmap(calculate_likelihood_real_scales, inputs)



#####################################################
### Re-Package Results


### Get average of average likelihoods
### (needed for accurately normalizing scale likelihoods to get probabilities)
def get_average_likelihood(n, octave=False, redo=False, imax=1250):
    txt = 'octave' if octave else str(imax)
    path_ave = PATH_COMP.joinpath('CostFn/AverageLikeNorm/', f"{txt}_{n}.npy")
    if path_ave.exists() and not redo:
        return 

    if not octave:
        path_list = sorted(Path(PATH_COMP.joinpath('CostFn/Normalization/')).glob(f"{imax}_{n}_*npy"))
    else:
        path_list = sorted(Path(PATH_COMP.joinpath('CostFn/Normalization/')).glob(f"octave_{n}_*npy"))

    cost_model = np.mean([np.load(p) for p in path_list], axis=0)
    np.save(path_ave, cost_model)


### Read all of the estimates of average likelihoods, and combine into
### one highly accurate estimate of the average likelihoods.
def run_all_average_likelihoods(redo=False, imax=1250):
    octave = [False, True]
    for n in N_INTS:
        for o in octave:
            get_average_likelihood(n, o, redo, imax)


### Unpack the values in the massive likelihood array, into smaller,
### organized chunks:
###     likelihood array :: (n_beta, n_model)
###     new data structure :: (n_model_type, n_beta, *model_params)
def unpack_likelihood_arrays(like, includes_scales=False):
    info_names = CF.info_model_names()
    params = CF.get_model_params()

    if not includes_scales:
        shape1 = (N_BETA, 2, 2, len(NAMES))
        shape2 = (N_BETA, 2, 2 * len(CF.M_ARR) + len(CF.W_ARR))
    else:
        n_scale = like.shape[-1]
        shape1 = (N_BETA, 2, 2, len(NAMES), n_scale)
        shape2 = (N_BETA, 2, 2 * len(CF.M_ARR) + len(CF.W_ARR), n_scale)
    n_hi = np.prod(shape1[1:4])

    like_hi = like[:,:n_hi].reshape(shape1)
    like_info = like[:,n_hi:].reshape(shape2)
    like_dict = {}

    cost_function_names = list(params.keys())
    start_HI = 0
    start_I = 0
    for name in cost_function_names:
        param_shape = tuple(len(x) for x in params[name])
        n_param = np.prod(param_shape)
        if name[0] in 'HI':
            end = start_HI + n_param
            p0 = like_hi[:,:,:,start_HI:end]
            params[name] = [['I', 'A'], ['CF1', 'CF2']] + params[name]

            if not includes_scales:
                like_dict[name] = p0.reshape(shape1[:-1] + param_shape)
            else:
                like_dict[name] = p0.reshape(shape1[:-2] + param_shape + (n_scale,))
            start_HI = end
        else:
            end = start_I + n_param
            p0 = like_info[:,:,start_I:end]
            params[name] = [['I', 'A']] + params[name]

            if not includes_scales:
                like_dict[name] = p0.reshape(shape2[:-1] + param_shape)
            else:
                like_dict[name] = p0.reshape(shape2[:-2] + param_shape + (n_scale,))
            start_I = end

    return like_dict, params


def unpack_likelihood_arrays_gen(n_ints, base='AverageLikeNorm', redo=False, imax=1250, includes_scales=False):
    octave = [False, True]
    for n, o in product(n_ints, octave):
        txt = 'octave' if o else str(imax)
        path_like = PATH_COMP.joinpath("CostFn", base, f"{txt}_{n}.npy")
        like = np.load(path_like)
        like_dict = unpack_likelihood_arrays(like, includes_scales=includes_scales)[0]

        name = path_like.stem
        for k, v in like_dict.items():
            path_model = PATH_COMP.joinpath("CostFn", f"{base}_Model", f"{k}_{name}.npy")
            if not path_model.exists() or redo:
                np.save(path_model, v)


def unpack_likelihood_arrays_norm(n_ints, base='AverageLikeNorm', redo=False, imax=1250):
    n_ints = np.arange(2, 10)
    unpack_likelihood_arrays_gen(n_ints, redo=redo)


def unpack_likelihood_arrays_real_scales(redo=False, imax=1250):
    n_ints = np.arange(2, 26)
    unpack_likelihood_arrays_gen(n_ints, base="EmpiricalLike", includes_scales=True, redo=redo)



#####################################################
### Scale Probabilities


def normalize_likelihood(model, octave, n, redo=False, imax=1250):
    txt = 'octave' if octave else str(imax)
    path_prob = PATH_COMP.joinpath("CostFn", f"ScaleProb", f"{model}_{txt}_{n}.npy")
    if path_prob.exists() and not redo:
        return

    path_like = PATH_COMP.joinpath("CostFn", f"EmpiricalLike_Model", f"{model}_{txt}_{n}.npy")
    like = np.load(path_like)

    path_norm = PATH_COMP.joinpath("CostFn", f"AverageLikeNorm_Model", f"{model}_{txt}_{n}.npy")
    norm = np.load(path_norm)

    try:
        prob = like / norm.reshape(norm.shape + (-1,))
        np.save(path_prob, prob)
    except Exception as e:
        print(path_like)
        print(path_norm)
        print(e, '\n')


def normalize_all_likelihoods(redo=False):
    models = list(CF.get_model_params().keys())
    octave_list = [False, True]
    with Pool(N_PROC) as pool:
        inputs = [(m, o, n, redo) for m, n, o in product(models, N_INTS, octave_list)]
        pool.starmap(normalize_likelihood, inputs)


#####################################################
### Model Probabilities


def model_probability(model, n, wtype='between'):
    # Load weights
    df = scales_io.load_all_data()

    if wtype == 'between':
        weights = WS.get_weights_by_region(df)[n]
        samp_crit = np.arange(0, 105, 5)

    elif wtype == 'within':
        weights = WS.get_weights_within_regions(df)[n]
        samp_crit = np.array(sorted(df.Region.unique()))

    elif wtype == 'overall':
        weights = WS.get_weights_by_region_overall(df)[:,:,df.n_ints==n]
        samp_crit = [f"o{x}" for x in np.arange(0, 105, 5)]

    # Load scale probabilities
    # Load non-octave (imax=1250) probabilities for Vocal / Instrumental scales,
    is_theory = weights[3,1] > 0
    path_prob = PATH_COMP.joinpath("CostFn", f"ScaleProb", f"{model}_1250_{n}.npy")
    prob = np.load(path_prob)
    shape = prob.shape
    prob = prob.reshape(-1, shape[-1])

    # Load octave probabilities for Theory scales.
    path_prob = PATH_COMP.joinpath("CostFn", f"ScaleProb", f"{model}_octave_{n}.npy")
    prob[:,is_theory] = np.load(path_prob).reshape(-1, shape[-1])[:,is_theory]
    prob = prob.reshape(shape)


    # Calculate model probabilities
    # (actually its the log-likelihood ratio of model over null model, but we'll write "mprob")
    shape1 = prob.shape
    shape2 = weights.shape
    prob_reshaped = prob.reshape(1, -1, shape1[-1])
    weight_sum_reshaped = np.sum(weights, axis=2).reshape(-1, 1, 1)
    weights_reshaped = weights.reshape(-1, 1, shape2[-1]) / weight_sum_reshaped

    mprob = np.sum(np.log(prob_reshaped) * weights_reshaped, axis=2)
    mprob = mprob.reshape(shape2[:-1] + shape1[:-1])

    # Remove infinite values
    mprob[~np.isfinite(mprob)] = np.nan

    scale_type = ['All', 'Vocal', 'Instrumental', 'Theory']
    for i, st in enumerate(scale_type):
        for j, sc in enumerate(samp_crit):
            path_mp = PATH_COMP.joinpath("CostFn", f"ModelProb", f"{st}_{sc}_{model}_{n}.npy")
            np.save(path_mp, mprob[i,j])


def run_all_model_prob(redo=False):
    models = list(CF.get_model_params().keys())
#   weight_type = ['between', 'within', 'overall']
#   weight_type = ['overall']
    weight_type = ['between', 'within']
    with Pool(N_PROC) as pool:
        inputs = [(m, n, w) for m, n, w in product(models, N_INTS, weight_type)]
        pool.starmap(model_probability, inputs)



#####################################################
### Integrate Models with different N


### Model & Scale Probabilities have been defined separately for different values of N.
### This function integrates the probabilities to find the best model.
###     model : model_name (e.g. "H_GFO_20")
###     st : scale_type
###     sc : sampling_criterion
### Integration can occur in the following modes:
###     A: no parameters vary with N
###     B: beta varies with N (so far the default option)
###     C: all parameters vary with N
###     DA: use provided model indices, beta DOES NOT vary with N
###     DB: use provided model indices, beta DOES vary with N
### By default returns the mean log-likelihood ratio (LLR; model to null) per scale (norm=True),
###     but can also return the total log-likelihood ratio.
def best_model_across_n(model, st, sc, mode='A', idx0=None, norm=True, sep_n=False, fix_vals=True, po_only=False):

    # Load weights
    df = scales_io.load_all_data()
    if isinstance(sc, int):
        weights = WS.get_weights_by_region(df)
    elif sc[0] == 'o':
        weights = WS.get_weights_by_region_overall(df)
    else:
        weights = WS.get_weights_within_regions(df)
    ist, insamp = get_weight_index(st, sc)

    if "D" in mode:
        if isinstance(idx0, type(None)):
            idx0 = get_best_param_idx()[st][model]

    # Container for saving indices of the best model(s)
    idx_best = []

    # Container for saving results per N
    best_n = []

    # Load models for each N, and depending on the mode, choose
    # the best models out of a set of parameters
    for i, n in enumerate(N_INTS):
        # Load model LLR
        path_mp = PATH_COMP.joinpath("CostFn", f"ModelProb", f"{st}_{sc}_{model}_{n}.npy")
        mprob = np.load(path_mp)


        # A hacky way to select certain parameters that perform best, or for testing purposes
        if fix_vals and model[0] in 'HI':
            mprob = mprob[:,:,[0]]

        # A hacky way to select certain parameters that perform best, or for testing purposes
        if fix_vals and model == 'C_CI':
            mprob = mprob[:,[1]]
            mprob = mprob[:,:,[0]]
#           mprob = mprob[:,:,:,[7]]

        # A hacky way to select certain parameters that perform best
        if fix_vals and model == 'C_NI':
            mprob = mprob[:,[1]]

        # Load sum of weights
        if isinstance(weights, dict):
            wsum = weights[n][ist,insamp,:].sum()
        else:
            wsum = weights[ist,insamp,df.n_ints==n].sum()

        if wsum == 0:
            best_n.append(np.nan)
            continue

        if mode == "B":
            # Optimize over beta
            mprob = np.nanmax(mprob, axis=0)

        elif mode == "C":
            # Optimize over all parameters (just for laffs, see what happens)
            idx = np.unravel_index(np.nanargmax(mprob), mprob.shape)
            idx_best.append(idx)
            mprob = mprob[idx] 

        elif mode == "DA":
            # All the parameter indices are provided in this case
            mprob = mprob[idx0]

        elif mode == "DB":
            # Merge the provided model indices with all possible beta values
            mprob = mprob[(range(N_BETA),) + idx0]
            # and get the optimum beta value
            idx = np.nanargmax(mprob)
            idx_best.append((idx,) + idx0)
            mprob = mprob[idx] 

        if isinstance(mprob, (float)):
            best_n.append(mprob)
        else:
            best_n.append(np.nanmax(mprob))

        if "prob_overall" not in locals():
            prob_overall = mprob * wsum
        else:
            prob_overall = prob_overall + mprob * wsum

    if norm:
        if isinstance(weights, dict):
            wsum = np.sum([weights[n][ist,insamp,:].sum() for n in N_INTS])
        else:
            wsum = weights[ist,insamp].sum()
        prob_overall = prob_overall / wsum

    if po_only:
        return prob_overall

    # Depending on the mode (if there are any parameters not yet optimized)
    # optimize the remaining parameters
    idx = np.unravel_index(np.nanargmax(prob_overall), prob_overall.shape)
    if mode == "A":
        # Optimize over all parameters after summing across N
        idx_best.append(idx)
    elif mode == "B":
        # Optimize over all parameters except beta
        idx_best.append(idx)
    elif mode == "DA":
        idx_best.append(idx0)

    if not isinstance(prob_overall, float):
        prob_overall = prob_overall[idx] 

    # Return the results separately for different values of N
    if sep_n:
        return prob_overall, idx_best, best_n

    return prob_overall, idx_best


def print_best_models(mode='A', nsamp_max=20, i0=[3]):
    names = updated_param_names()
    params = get_updated_model_param_vals()
    for st in SCALE_TYPE:
        for model in MAIN_MODELS[i0]:
            po, idx = best_model_across_n(model, st, nsamp_max, mode=mode)
            print(st, model, po)
            print(idx[0])
            if mode == 'A':
                p_txt = f"beta = {BETA_ARR[idx[0][0]]}"
                idx0 = idx[0][1:]
            elif mode == 'B':
                p_txt = ""
                idx0 = idx[0]
            for i, n, p in zip(idx0, names[model], params[model]):
                p_txt = p_txt + f", {n} = {p[i]}"
            print(p_txt, '\n')
            


#####################################################
### Various functions (should probably go to utils)


def updated_param_names(exclude_M=True):
    names = CF.get_param_names()
    for k, v in names.items():
        if exclude_M:
            if k[0] in 'HI':
                names[k] = ["I/A", "CF"] + v
            else:
                names[k] = ["I/A"] + v
        else:
            if k[0] in 'HI':
                names[k] = ["I/A", "CF"] + v
            else:
                names[k] = ["I/A"] + v
    return names


def get_updated_model_param_vals(exclude_M=True):
    params = CF.get_model_params()
    for name in params.keys():
        if exclude_M:
            if name[0] in 'HI':
                params[name] = [['I', 'A'], ['CF1', 'CF2']] + params[name]
            else:
                params[name] = [['I', 'A']] + params[name]
        else:
            if name[0] in 'HI':
                params[name] = [['I', 'A'], ['CF1', 'CF2'], CF.M_ARR] + params[name]
            else:
                params[name] = [CF.M_ARR, ['I', 'A']] + params[name]
    return params


def get_weight_index(st, nsamp_max):
    scale_type = np.array(['', 'Vocal', 'Instrumental', 'Theory'])
    n_max = np.arange(0, 105, 5)
    ist = np.where(scale_type == st)[0][0]
    if isinstance(nsamp_max, int):
        insamp = np.where(n_max == nsamp_max)[0][0]

    elif isinstance(nsamp_max, str):
        if nsamp_max[0] == 'o':
            insamp = np.where(n_max == int(nsamp_max[1:]))[0][0]
        else:
            df = scales_io.load_all_data()
            regions = np.array(sorted(df.Region.unique()))
            insamp = np.where(regions == nsamp_max)[0][0]
    return ist, insamp


def get_model_score(model):
    idx = np.where(NAMES == model)[0]
    if not len(idx):
        print(f"Model name '{model}' not found")
    else:
        return SCORE[idx[0]]


def get_best_param_idx():
    idx = {'':{'H_GP': (1, 0, 0, 0),
               'H_GFO': (1, 0, 0),
               'H_HP': (1, 0, 4, 5),
               'C_CI': (1, 0, 1),
               'C_NI': (1, 0)}, 
           'Vocal':{'H_GP': (1, 0, 0, 11),
                    'H_GFO': (1, 0, 9),
                    'H_HP': (1, 0, 0, 2),
                    'C_CI': (1, 0, 2),
                    'C_NI': (1, 2)}, 
           'Instrumental':{'H_GP': (1, 0, 0, 15),
                           'H_GFO': (1, 0, 7),
                           'H_HP': (1, 0, 0, 2),
                           'C_CI': (1, 0, 4),
                           'C_NI': (1, 3)}, 
           'Theory':{'H_GP': (1, 0, 0, 0),
                     'H_GFO': (1, 0, 1),
                     'H_HP': (1, 0, 36, 0),
                     'C_CI': (1, 0, 0),
                     'C_NI': (1, 0)}} 
    return idx


def model_name_examples():
    names = ["A_1_H_GP_1_20", "A_1_H_GFO_20", "A_1_H_HP_3_1", "A_1_H_HP_39_1",
             "C_CI_A_abs_1.0", "C_NI_A_20", "C_NI_I_20"]
    return names


def all_model_names():
    info_names = CF.info_model_names()
    names0 = [f"{a}_{b}_{n}" for a in 'IA' for b in '12' for n in NAMES]
    names0.extend(info_names)
    return np.array(names0)


if __name__ == "__main__":
    # Run the following if the normalization constants are not downloaded (or preprocessed)
    if 0:
        # Estimate average likelihoods for all models
        estimate_average_run_all()
        run_all_average_likelihoods()

        # Restructure likelihood arrays
        unpack_likelihood_arrays_norm(redo=False)

    # Get likelihoods for empirical scales, for all models
    if 1: 
        run_all_likelihood_real_scales(redo=True)
        unpack_likelihood_arrays_real_scales(redo=True)

    # Convert likelihoods into scale probabilities
    if 1:
        normalize_all_likelihoods(redo=False)

    # Compare models
    if 1:
        run_all_model_prob(redo=False)





