"""
This module, "interval_stats.py", contains code for estimating the degree
of randomness in interval statistics.

See the bottom of the code for details of the main functions to run,
in order to reproduce the work.

"""
import os
from pathlib import Path
import pickle

from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, lognorm, norm, binom, entropy, ks_1samp, false_discovery_control, mstats
from sklearn.manifold  import TSNE
from sklearn.preprocessing import StandardScaler


import cost_functions as CF
import evaluate_models as EM
from scales_config import *
import scales_io
import scales_utils as utils
import weighted_sampling as WS


####################################################
### Interval statistics
### Estimating non-randomness of empirical interval populations


### Count how many times intervals are found, and compare this with
### what one would expect from one of the random processes
### (drawn from a lognormal distribution, shuffled empirical scales,
### or resampled from the empirical step distribution)
def get_int_prob_lognorm(df, ysamp='AllInts', xsamp='', s=5, mode='', ai_alg='first'):
    # Shuffling means taking the step intervals, and rearranging them
    if mode == 'shuffle':
        if xsamp != '':
            ints = utils.sample_shuffled_scales(df, xsamp, min(s, len(df)))
        else:
            ints = utils.return_shuffled_scales(df)

        if ysamp == 'Scale':
            # Need to throw away the final interval, because this is never changed
            # by shuffling, which results in biased sampling.
            # Therefore, we throw away the final interval, and instead of having biased
            # results, we just have lower significance for high intervals
            Y = [x for i in ints for x in np.cumsum(i)[:-1]]
        elif ysamp == 'AllInts':
            Y = [x for i in ints for x in CF.get_all_ints(i, ai_alg)]

    # Resampling is generating entirely new sets of step sizes, to match the
    # number of step sizes in the original sample
    elif mode == 'resample':
        if xsamp != '':
            idx = utils.sample_df_index(df, xsamp, s)
            alt_df = utils.create_new_scales(df.loc[idx], 1)[0]
        else:
            alt_df = utils.create_new_scales(df, 1)[0]

        if ysamp == 'Scale':
            Y = np.array([x for y in alt_df['Ints'].values for x in np.cumsum(y)])
        elif ysamp == 'AllInts':
            Y = np.array([x for y in alt_df['Ints'].values for x in CF.get_all_ints(y, ai_alg)])

    else:
        if xsamp != '':
            Y = utils.sample_df_value(df, ysamp, xsamp, s)
        else:
            Y = [x for y in df[ysamp] for x in y]

    bins = np.arange(10, 1810, 20)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    try:
        # Get maximum-likelihood lognormal distribution
        shape, loc, scale = [0.93, -45.9, 605.4]
        params = lognorm.fit(Y, loc=loc, scale=scale)

        # Get probability of finding interval in each bin
        bin_prob = np.diff(lognorm.cdf(bins, *params))

        # Count intervals in each bin
        count = np.histogram(Y, bins=bins)[0]
        N = count.sum()

        # Get binomial probability that observed counts (or fewer) would be
        # sampled from the maximum-likelihood lognormal distribution
        prob_less_than = binom.cdf(count, N, bin_prob)
        return [count, np.ones(count.size)*N, prob_less_than]

    except Exception as e:
        print(e)
        print(ysamp, xsamp, s, mode)
        return [np.ones(X.size)*np.nan] * 3


def generate_int_prob_lognorm(df, xsamp, ysamp, nsamp, mode='', nrep=1000, ai_alg='first'):
    for y in ysamp:
        for n in nsamp:
            for i in range(nrep):
                yield df, y, xsamp, n, mode, ai_alg


def boot_int_prob_lognorm(df, path, nrep=1000, redo=False, mode='', xsamp='', ai_alg='first'):
    if path.exists() and not redo:
        return np.load(path)
    else:
        print(path)
        df = df.loc[:, ['Ints', 'Scale', 'AllInts', 'Region']]
        bins = np.arange(10, 1810, 20)
        ysamp = ['Scale', 'AllInts']

        # Only sample at different rates if sampling across regions,
        # otherwise just count
        if xsamp == 'Region':
            nsamp = np.arange(5, 55, 5)
        elif xsamp == '':
            nsamp = [0]

        shape = (len(ysamp), len(nsamp), nrep, 3, bins.size-1)
        with Pool(scales_io.N_PROC) as pool:
            inputs = generate_int_prob_lognorm(df, xsamp, ysamp, nsamp, mode, nrep, ai_alg)
            res = np.array(pool.starmap(get_int_prob_lognorm, inputs)).reshape(shape)
            np.save(path, res)
        return res


def boot_int_prob_lognorm_all(redo=True, nrep=1000, min_reg=10):
    df = scales_io.load_all_data()
    df = df.loc[(df.n_ints>=1)]
    scale_type = ['Vocal', 'Instrumental', 'Theory']
    regions = np.array(sorted(df.Region.unique()))
    path_data = scales_io.PATH_RES.joinpath("IntervalStats")
    
    alg = {'Theory':'circular'}
    df['AllInts'] = [CF.get_all_ints(x, alg.get(y, 'first')) for x, y in zip(df['Ints'], df['ScaleType'])]

    for st in scale_type:
        ai_alg = alg.get(st, 'first')

        # Sample across regions first
        idx = (df.ScaleType==st)

        path = path_data.joinpath(f"int_prob_lognorm_{st}.npy")
        _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, nrep=nrep, xsamp='Region', ai_alg=ai_alg)

        path = path_data.joinpath(f"int_prob_lognorm_{st}_shuffle.npy")
        _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, mode="shuffle", nrep=nrep, xsamp='Region', ai_alg=ai_alg)

        path = path_data.joinpath(f"int_prob_lognorm_{st}_resample.npy")
        _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, mode="resample", nrep=nrep, xsamp='Region', ai_alg=ai_alg)

        for reg in regions:
            idx = (df.ScaleType==st) & (df.Region==reg)
            if np.sum(idx) < min_reg:
                continue

            path = path_data.joinpath(f"int_prob_lognorm_{st}_{reg}.npy")
            _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, nrep=1, ai_alg=ai_alg)

            path = path_data.joinpath(f"int_prob_lognorm_{st}_{reg}_shuffle.npy")
            _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, mode="shuffle", nrep=1, ai_alg=ai_alg)

            path = path_data.joinpath(f"int_prob_lognorm_{st}_{reg}_resample.npy")
            _ = boot_int_prob_lognorm(df.loc[idx], path, redo=redo, mode="resample", nrep=1, ai_alg=ai_alg)



def get_significant_intervals(st, reg, i1, i2, mode):
    path_data = scales_io.PATH_RES.joinpath("IntervalStats")
    reg = f"_{reg}" if len(reg) else reg

    count = np.load(path_data.joinpath(f"int_prob_lognorm_{st}{reg}.npy"))[i1,i2,:,0]
    Narr = np.load(path_data.joinpath(f"int_prob_lognorm_{st}{reg}.npy"))[i1,i2,:,1]

    count_t2 = np.load(path_data.joinpath(f"int_prob_lognorm_{st}{reg}_{mode}.npy"))[i1,i2,:,0]
    Narr_t2 = np.load(path_data.joinpath(f"int_prob_lognorm_{st}{reg}_{mode}.npy"))[i1,i2,:,1]
    prob_t2 = np.mean(count_t2 / Narr_t2, axis=0)

    prob_less_t2 = np.array([binom.cdf(count[i], Narr[i], prob_t2) for i in range(len(count))])
    prob_obs_t2 = np.min([prob_less_t2, 1 - prob_less_t2], axis=0)
    is_less = prob_less_t2.mean(axis=0) < 0.5
    return false_discovery_control(mstats.gmean(prob_obs_t2, axis=0)), is_less


def get_significant_intervals_st(i1=1, i2=4, mode='resample', reg=''):
    scale_type = ['Vocal', 'Instrumental', 'Theory']
    prob_list = []
    return np.array([get_significant_intervals(st, reg, i1, i2, mode) for st in scale_type])


def get_sig_int_region(df, mode='resample'):
    df = df.copy()
    df = utils.rename_regions2(df)
    scale_type = ['Vocal', 'Instrumental', 'Theory']
    regions = np.array(sorted(df.Region.unique()))
    out = {st: {} for st in scale_type}
    for st in scale_type:
        for reg in regions:
            try:
                out[st][reg] = get_significant_intervals(st, reg, 1, 0, mode)
            except Exception as e:
                pass
    return out


####################################################
### Scale costs
### Calculate cost functions of real and random scales


def best_model_params():
    params = {'H_GP': [4, 3, 0],
              'H_GFO': [54,  52, 50],
              'H_HP': [66, 62, 82],
              'C_CI': [(0, 7), (0, 7), (0, 7)],
              'C_NI': [6, 5, 0]} # [9, 9, 9]
    return params


def real_scale_costs(imax=1250, redo=False):
    path = scales_io.PATH_COMP.joinpath("CostFn", "empirical_cost_best_models.csv")
    if path.exists() and not redo:
        return pd.read_csv(path) 

    df = scales_io.load_all_data()
    df = df.loc[(df.n_ints>=1)&(df.n_ints<=9)]
    n_ints = np.arange(2, 10)
    scale_type = ['Vocal', 'Instrumental', 'Theory']
    ai_alg = ['first', 'first', 'circular']
    names = ['H_GP', 'H_GFO', 'H_HP', 'C_CI', 'C_NI']
    params = best_model_params()

    data = []
    for i, st in enumerate(scale_type):
        for j, n in enumerate(n_ints):
            ints = np.array([x for x in df.loc[(df.n_ints==n)&(df.ScaleType==st), 'Ints']]).astype(int)
            if not len(ints):
                continue

            regions = df.loc[(df.n_ints==n)&(df.ScaleType==st), 'Region']

            all_ints = CF.get_all_ints(ints, ai_alg[i])
            all_ints[all_ints >= imax] = 0

            costs = []
            for name in names:
                if name[0] == 'H':
                    cost = np.nanmean(EM.SCORE[params[name][i]][all_ints], axis=1)
                elif name == 'C_CI':
                    cost = CF.template_function_many(ints)[params[name][i]]
                elif name == 'C_NI':
                    cost = CF.number_unique_intervals_many(ints)[params[name][i]]

                for k, (c, r) in enumerate(zip(cost, regions)):
                    ID = f"{n}_{k:03d}"
                    data.append((st, n, ID, name, c, r))

    df_cost = pd.DataFrame(data=data, columns=['ScaleType', 'n_ints', 'ID', 'model', 'cost', 'Region'])
    df_cost.to_csv(path)
    return df_cost


def sample_real_scale_costs(df=None, st='Vocal', nsamp=20, cost_only=False):
    if isinstance(df, type(None)):
        df = real_scale_costs()

    scale_type = ['Vocal', 'Instrumental', 'Theory']
    n_ints = np.arange(2, 10)
    df_out = []
    id_list = []
    for j, n in enumerate(n_ints):
        idx = (df.n_ints==n) & (df.ScaleType==st) & (df.model=='H_GP')
        id_list.extend(list(utils.sample_df_value2(df.loc[idx], 'ID', 'Region', nsamp)))
    df = df.loc[(df.ScaleType==st) & (df.ID.isin(id_list))]
    if cost_only:
        key = {m: i for i, m in enumerate(MAIN_MODELS)}
        df['model_order'] = [key.get(m) for m in df['model']]
        return df.sort_values(by=['model_order', 'ID'])['cost'].values.reshape(5, -1)
    else:
        return df


def random_scale_costs(df=None, st='Vocal', nsamp=20, imax=1250):
    if isinstance(df, type(None)):
        df = real_scale_costs()
    int_set, prob = scales_io.load_int_dist()
    df = sample_real_scale_costs(df, st, nsamp)
    n_ints = df.loc[(df.ScaleType==st)&(df.model=='H_GP'), 'n_ints']

    names = ['H_GP', 'H_GFO', 'H_HP', 'C_NI']
    octave = st == 'Theory'
    ai_alg = 'circular' if octave else 'first'

    ints_list = []
    for n in n_ints:
        ints = np.random.choice(int_set, p=prob, size=n)
        if octave:
            ints = ints / (np.sum(ints) / 1200)
        ints_list.append(np.round(ints).astype(int))

    params = best_model_params()
    i0 = {'Vocal':0, 'Instrumental':1, 'Theory':2}[st]
    all_ints = []
    for i in ints_list:
        ai = CF.get_all_ints(i, ai_alg)
        ai[ai >= imax] = 0
        all_ints.append(ai)

    cost_list = []
    for name in names:
        if name[0] == 'H':
            cost = [np.nanmean(EM.SCORE[params[name][i0]][ai]) for ai in all_ints]
        elif name == 'C_CI':
            cost = np.array([CF.template_function_single(i, 1, 'abs') for i in ints_list])
        elif name == 'C_NI':
            cost = CF.number_unique_intervals_many(ints_list)[params[name][i0]]
        cost_list.append(cost)

    return np.array(cost_list)


def boot_scale_costs(redo=False, nrep=1000, nsamp=20):
    df = real_scale_costs()
    scale_type = ['Vocal', 'Instrumental', 'Theory']

    path_data = scales_io.PATH_RES.joinpath("IntervalStats")
    path = path_data.joinpath(f"scale_cost_ran_{nsamp}.pkl")
    if path.exists() and not redo:
        ran_cost = pickle.load(open(path, 'rb'))
    else:
        ran_cost = {}
        for st in scale_type:
            print(st)
            with Pool(scales_io.N_PROC) as pool:
                inputs = [(df, st, nsamp)] * nrep
                ran_cost[st] = np.array(pool.starmap(random_scale_costs, inputs, 5))
#       ran_cost = {st: np.array([random_scale_costs(df, st, nsamp) for _ in range(nrep)]) for st in scale_type}
        pickle.dump(ran_cost, open(path, 'wb'))

    
    path = path_data.joinpath(f"scale_cost_real_{nsamp}.pkl")
    if path.exists() and not redo:
        real_cost = pickle.load(open(path, 'rb'))
    else:
        real_cost = {}
        for st in scale_type:
            print(st)
            with Pool(scales_io.N_PROC) as pool:
                inputs = [(df, st, nsamp, 1)] * nrep
                real_cost[st] = np.array(pool.starmap(sample_real_scale_costs, inputs, 5))
#       real_cost = np.concatenate([sample_real_scale_costs(df, st, nsamp, cost_only=1) for _ in range(nrep)], axis=1)
        pickle.dump(real_cost, open(path, 'wb'))

    return real_cost, ran_cost


####################################################
### Scale degree KDE
### Generate kernel density estimates for empirical scale degree distributions


def scale_degree_distributions(nmax=20, redo=False):
    df = scales_io.load_all_data()
    n_ints = np.arange(2, 26)
    for i, st in enumerate(SCALE_TYPE):
        for j, n in enumerate(n_ints):
            xmax = 300 * n
            if not np.any((df.ScaleType==st)&(df.n_ints==n)):
                continue
            path_kde = PATH_COMP.joinpath("ScaleDegreeDist", f"{st}_{n}_{nmax}.npy")
            if not path_kde.exists() or redo:
                X, Y = WS.get_weighted_kde(df, st, n, nmax, xmax=xmax)
                np.save(path_kde, np.array([X, Y]).T)



if __name__ == "__main__":

    # Calculate non-randomness of intervals
    boot_int_prob_lognorm_all(redo=1)

    # Calculate cost functions of real and random scales
    for nsamp in range(5, 55):
        boot_scale_costs(1, 1000, nsamp)

    # Generate kernel density estimates for empirical scale degree distributions
    scale_degree_distributions()



