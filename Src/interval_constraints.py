"""
This module, "interval_constraints.py", contains code for modelling
the distribution of step sizes.

See the bottom of the code for details of the main functions to run,
in order to reproduce the work.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, linregress
from scipy.optimize import brute, minimize
import seaborn as sns

from scales_config import *
import scales_utils as utils


##################################################################
### I/O

### Expand scales dataframe so that it is arranged at the level
### of step intervals
def get_interval_df(df):
    col = ['Ints', 'ScaleType', 'n_notes', 'Region', 'Method']
    newcol = ['Int', 'ScaleType', 'n_ints', 'Region', 'Method']
    data = {c:[] for c in newcol}
    for i in df.index:
        ints, st, n, reg, method = df.loc[i, col]
        n -= 1
        data['Int'].extend(list(ints / 100))
        data['ScaleType'].extend([st]*n)
        data['n_ints'].extend([n]*n)
        data['Region'].extend([reg]*n)
        data['Method'].extend([method]*n)
    return pd.DataFrame(data)


##################################################################
### Statistics of step size distributions,
### for empirical, Null, and model-generated populations of scales


def estimate_stat(step, N, n_rep=10000, fn=np.min):
    return np.mean(fn(np.random.choice(step, replace=True, size=N*n_rep).reshape(n_rep,N), axis=1))


def estimate_stat_limit(step, N, n_rep=10000, fn=np.min, scale_range=1200, octave=False):
    ints = np.random.choice(step, replace=True, size=N*n_rep).reshape(n_rep,N)
    if octave:
        ints = ints / ints.sum(axis=1).reshape(-1, 1) * scale_range

    fn_res = fn(ints, axis=1)
    idx = ints.sum(axis=1) <= scale_range
    return np.mean(fn_res[idx])


### Step interval probability distribution as a function of scale size and number of steps,
###     which assumes that all scales are equally likely, for those scales
###     where the sum of the steps is less than or equal to scale_range
### X           :: interval grid
### scale_range :: maximum allowed sum of intervals
### n           :: number of intervals
def get_int_prob_null(X, scale_range, n):
    return np.exp((n - 1) * np.log(scale_range - X) + n - n * np.log(scale_range))


### Get the expected value of fn(scale) for a set of randomly generated
### scales, assuming all scales are equally likely.
### The default case, "fn=np.min", results in the expected value of the minimum
### step per scale
def get_null_stat(scale_range, N, n_rep=1000, fn=np.min, octave=False):
    if octave:
        if scale_range != 1200:
            print("Warning! Scale range is not compatible with an octave")
    X = np.arange(1, scale_range).astype(float)
    prob = get_int_prob_null(X, scale_range, N)
    prob = prob / prob.sum()
    return np.mean(fn(np.random.choice(X, replace=True, p=prob, size=N*n_rep).reshape(n_rep,N), axis=1))

def change_scale(x0, A, dx_max=200):
    scale = np.cumsum(x0)
    scale += (np.random.rand(len(x0)) - 0.5) * dx_max
    scale = scale % A
    ints = np.diff([0] + sorted(scale))
    return ints


def get_expected_minmax_mc(A, N, fn=np.min, tol=0.1):
    x0 = np.random.rand(N)
    x0 = x0 / np.sum(x0) * np.random.rand() * A
    total = fn(x0)
    count = 1
    nstep = 10000
    diff = 10.**6
    estimate = diff
    while diff > tol:
        x0 = change_scale(x0, A)
        total += fn(x0)
        count += 1
        if count % nstep == 0:
            new_estimate = total / count
            diff = abs(new_estimate - estimate)
            estimate = new_estimate
            print(A, N, count, estimate, diff)
    return estimate


def estimate_minmax():
    N = np.arange(2, 12)
    emin = np.array([get_expected_minmax_mc(1200, n, np.min, 0.0001) for n in N])
    emax = np.array([get_expected_minmax_mc(1200, n, np.max, 0.0001) for n in N])
    np.save('../Precompute/expected_min.npy', emin/1200)
    np.save('../Precompute/expected_max.npy', emax/1200)


### Variable range null model
def null_var_range(df):
    n_notes = np.arange(2, 12)
    range_range = np.arange(300, 2300, 100)
    bins = np.arange(250, 2300, 100)
    null_range = np.array([[get_null_stat(r, n-1, 100) for n in n_notes] for r in range_range])

    hist = np.histogram(df.loc[df.ScaleType=='Vocal', 'scale_range'], bins=bins)[0]
    new_null = np.sum(null_range * (hist / hist.sum()).reshape(-1, 1), axis=0)

    plt.plot(n_notes, new_null)


##################################################################
### Melody (Interval Spacing + Motor Constraint) theory

### Probability that an interval will be accepted in a scale,
### based on interval spacing (IS)
def IS_probability_function(X, a, b):
    return norm.cdf(X / 2, 0, a)**b


### Probability that an interval will be accepted in a scale,
### based on motor constraints (MC)
def MC_probability_function(X, c):
    return np.exp(-X / c)


### Probability that an interval will be accepted in a scale,
### based on both theories (IS and MC)
def interval_probability_function(X, a, b, c):
    Y = IS_probability_function(X, a, b) * MC_probability_function(X, c)
    Y = Y / np.trapz(Y, X)
    return Y


def fit_fn(inputs, X, Y0, fn):
    Y = fn(X, *inputs)
    Y = Y / np.trapz(Y, X)
    return np.abs(Y - Y0).sum()


def fit_constraint_c(inputs, X, Y0, c=207.4):
    a, b = inputs
    Y = interval_probability_function(X, a, b, c)
    return np.abs(Y - Y0).sum()


### Fraction of scales that will be excluded as a function of
### the number of notes (n), and the scale range limit (r)
def get_frac_over(n, r):
    return 1 - norm.cdf(r, np.mean(step)*n, (np.std(step)**2 * n)**0.5)



##################################################################
### Fitting MC theory to independent melodic data

def fit_exp_fn(c, X, Y):
    Ypred = np.exp(-X / c)
    Ypred = Ypred / np.nansum(Ypred)
    return np.nansum(np.abs(np.log(Y) - np.log(Ypred)))


def fit_melodic_interval_distributions(int_prob=None, redo=False):
    path = PATH_DATA.joinpath("Melodies", 'melodic_interval_scaling_coef_per_corpus.npy')
    if path.exists() and not redo:
        return np.load(path)

    if isinstance(int_prob, type(None)):
        int_prob = np.load(PATH_DATA.joinpath("Melodies", 'melodic_interval_dist_per_corpus.npy'))

    X = np.arange(15)
    grad = []
    bounds = [(0.5, 10)]
    for prob in int_prob:
        prob = prob / np.nansum(prob)
        res = minimize(fit_exp_fn, [2.5], args=(X, prob), bounds=bounds)
        grad.append(res.x[0])
    grad = np.array(grad)
    np.save(path, grad)
        
    return grad
    

if __name__ == "__main__":
    estimate_minmax()
    



