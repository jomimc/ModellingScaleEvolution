"""
This module, "generate_scales.py", includes a class, "ScaleGenerator",
that can be used to generate populations of scales according to different
biases, and using different assumptions and sampling techniques.

The code can be repurposed using different models / cost functions,
and also using different Monte Carlo moves and probabilities.

This module also contains code for:
 > checking whether the generator produces converged distributions of scales
 > generating populations for hypothesis testing
 > evaluating the generated populations

See the bottom of the code for details of the main functions to run,
in order to reproduce the work.

"""
import argparse
import sys 
from multiprocessing import Pool
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

import cost_functions as CF
import evaluate_models as EM
from scales_config import *
import scales_io
import scales_utils as utils



##########################################################
### Generating scales

# Number of scales to process in one go
NMAX = 100000

### Default Monte Carlo parameters
P1a = 0.50           # Probability to generate new scale
P1b = 0.00           # Probability to sample from old scales
P1c = 0.50           # Probability to modify a scale
P2a = 0.20           # Probability to shuffle intervals
P2b = 0.80           # Probability to change a note
DX_MAX = 50         # Maximum value of note change
NSTEP = 1000

SCORE, NAMES = CF.precompute_all()
SCORE[:,0] = np.nan

### Monte Carlo simulation of scale selection.
### Given a model, its cost function, and parameters, this code generates
### multiple, different populations of scales. It starts with low bias strength, beta,
### and gradually increases the strength up to beta_max.
### The Monte Carlo moves (see lines 36-40) can be chosen with
### different probabilities, in a way that should only affect the rate of convergence,
### and not the final converged answer.
### One must define the size of the scale (n_ints), and this is a fixed parameter.
class ScaleGenerator(object):
    def __init__(self, n_ints, beta_max, cost_fn, params, **kwargs):

        # Positional arguments
        self.n_ints = n_ints            # Number of step intervals
        self.beta_max = beta_max        # Maximum value of beta
        self.cost_fn = cost_fn          # Cost function (string)
        self.cost_fn_params = params    # Parameters of the cost function (dict)

        # Default settings
        self.octave = False             # Octave scales will use "circular" all_ints alg instead of "first"
        self.beta_min = 1               # Minimum value of beta
        self.n_step = 10                # Number of steps from beta = 1 to beta = beta_max
        self.n_scales = 10000           # Number of scales generated in a population
        self.imax = 1250                # Maximum interval size used to calculate scores
        self.beta_arr_scale = 'linear'  # Linear vs log scale for beta values
        self.bias_ints = True           # Choose scales based on interval probability also?
        self.early_stop = True          # Stop when evaluation metric has reached a certain value?
        self.low_acc_rate = 0.010       # Stop simulation if acceptance rate is below this level
        self.single_beta = True         # Just run with a single value of beta
        self.uniform_max_int = 500      # Maximum step interval size from uniform distribution

        # Monte Carlo move probabilities (set to global variables)
        self.P1a = P1a
        self.P1b = P1b
        self.P1c = P1c
        self.P2a = P2a
        self.P2b = P2b

        self.dx_max = DX_MAX            # Maximum change of scale degree (global variable)
        self.n_report = 100000          # Number of iterations after which progress is reported

        # Load any setting changes
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initializing counters
        self.iter = 0
        self.int_populations = []
        self.int_pop_cost = []

        self.attempt = []
        self.accept = []

        # Pre-loading arrays
        if self.single_beta:
            self.beta_arr = np.array([beta_max])
            self.n_step = 1
        else:
            if self.beta_arr_scale == 'linear':
                self.beta_arr = np.linspace(self.beta_min, self.beta_max, self.n_step)
            else:
                self.beta_arr = np.logspace(np.log10(self.beta_min), np.log10(self.beta_max), self.n_step)

        if self.cost_fn[0] in 'HI':
            self.score = EM.get_model_score(self.cost_fn)
    
        # Load step interval probability distribution
        self.xgrid, self.prob = scales_io.load_int_dist()

    def run(self):
        for _ in range(self.n_step):
            self.generate_population()

        for attr in ['int_populations', 'int_pop_cost']:
            setattr(self, attr, np.array(getattr(self, attr)))


    def generate_population(self):
        print(f"Running iteration {self.iter}")
        beta = self.beta_arr[self.iter]
        self._update_move_prob()
        ints_set, ints_cost, ints_prob = [], [], []

        count = 0 
        attempt = np.zeros(4, float)
        accept = np.zeros(4, float)

        # First scale generated is always accepted
        old_cost = np.inf
        if self.bias_ints:
            old_int_prob = 0

        if self.iter != 0:
            oi_idx = list(range(len(self.int_populations[-1])))
            oi_prob = np.exp(-beta * self.int_pop_cost[-1])
            oi_prob = oi_prob / np.sum(oi_prob)

        while len(ints_set) < self.n_scales:
            cointoss = np.random.rand()

            # Only allow moves 1 and 2 if there have been no ints generated
            if len(ints_set) == 0:
                cointoss *= self.move_probA[1]

            if cointoss <= self.move_probA[0]:
                i1 = self.generate_new_ints()
                j = 0

            elif cointoss <= self.move_probA[1]:
                i1 = self.int_populations[-1][np.random.choice(oi_idx, p=oi_prob)]
                j = 1

            else:
                # Only choose the last ints, otherwise there is a history-dependent bias
                # towards scales that were generated first
                i1 = ints_set[-1]

                cointoss = np.random.rand()
                if cointoss < self.move_probB[0]:
                    i1 = self.shuffle_ints(i1)
                    j = 2
                else:
                    i1 = self.change_note(i1)
                    j = 3

            count += 1
            attempt[j] += 1

            # Upper bound on step size
            if np.any(i1 >= 800):
                continue

            # Calculate scale cost
            cost = self.get_cost(i1)

            # Calculate probability of acceptance
            if self.bias_ints:
                # Calculate interval probability
                int_prob = self.get_int_prob(i1)
                scale_prob = np.exp(-beta * (cost - old_cost)) * int_prob / old_int_prob
            else:
                scale_prob = np.exp(-beta * (cost - old_cost))

            cointoss = np.random.rand()
            # If accept, save ints, cost, prob, and update counters
            if (scale_prob > cointoss) or (beta == 0):
                ints_set.append(i1)
                ints_cost.append(cost)
                ints_prob.append(scale_prob)
                accept[j] += 1
                old_cost = cost
                if self.bias_ints:
                    old_int_prob = int_prob

            if (count % self.n_report == 0):
                print(f"Step {count}\n{len(ints_set)} scales accepted")
                print(f"Average cost = {np.mean(ints_cost)}")
                print(attempt)
                print(accept / attempt)
                if np.sum(attempt) / np.sum(accept) < self.low_acc_rate:
                    print("Simulation stopped early due to low acceptance!\n")
                    return

        self.int_populations.append(np.array(ints_set))
        self.int_pop_cost.append(np.array(ints_cost))

        self.accept.append(accept)
        self.attempt.append(attempt)


        self.iter += 1


    def get_cost(self, ints):
        modify_cost_fn = False
        params = {}
        for k, v in self.cost_fn_params.items():
            if k == 'I/A':
                if v == 'A':
                    if self.octave:
                        ints = CF.get_all_ints(ints, 'circular')
                    else:
                        ints = CF.get_all_ints(ints, 'first')

            elif k == 'CF':
                modify_cost_fn = True
                modify_fn = {'CF1': lambda x: -x,
                             'CF2': lambda x: (1 / (1 + x))}[v]
            else:
                params[k] = v


        if self.cost_fn[0] in 'HI':
            ints = np.round(ints.copy()).astype(int)
            ints[ints >= self.imax] = 0
            cost = np.nanmean(self.score[ints])

        elif self.cost_fn[:4] == 'C_CI':
            cost = CF.template_function_single(ints, **params)

        elif self.cost_fn[:4] == 'C_NI':
            cost = CF.number_unique_intervals(ints, **params)

        if modify_cost_fn:
            return modify_fn(cost)
        return cost


    def get_int_prob(self, ints):
        return np.prod(self.prob[np.round(ints).astype(int)])


    def generate_new_ints(self):
        if self.bias_ints:
            ints = np.random.choice(self.xgrid, p=self.prob, size=self.n_ints, replace=True)
        else:
            ints = np.random.rand(self.n_ints) * self.uniform_max_int

        if self.octave:
            ints = ints / (np.sum(ints) / 1200)
        return ints


    def change_note(self, ints):
        new_scale = np.array([-1])
        n = len(ints)
        while np.any(new_scale < 0):
            new_scale = np.cumsum([0] + list(ints))
            dx = (np.random.rand(1) - 0.5) * (2 * self.dx_max)
            if self.octave:
                if n > 2:
                    i = np.random.randint(n - 2) + 1 
                else:
                    i = 0
                new_scale[i] = new_scale[i] + dx
                new_scale = np.array(sorted(new_scale % 1200))
            else:
                i = np.random.randint(n - 1) + 1 
                new_scale[i] = new_scale[i] + dx
                new_scale = np.array(sorted(new_scale))
        return np.diff(new_scale)


    def shuffle_ints(self, ints):
        ints = np.copy(ints)
        np.random.shuffle(ints)
        return ints


    def reset(self, restart_from=0):
        if restart_from == 0:
            # Initializing counter from scratch
            self.iter = 0
            self.int_populations = []
            self.int_pop_cost = []

            self.attempt = []
            self.accept = []
        else:
            self.iter = restart_from
            self.int_populations = self.int_populations[:restart_from]
            self.int_pop_cost = self.int_pop_cost[:restart_from]

            self.attempt = self.attempt[:restart_from]
            self.accept = self.accept[:restart_from]


    def _update_move_prob(self):
        i = self.iter
        self.move_probA = np.array([0.5, 0.5, 1])
        self.move_probB = np.cumsum([P2a, P2b])


    def get_scale_kde_fn(self, ints):
        return gaussian_kde(np.cumsum(ints, axis=1).ravel(), bw_method=0.05)


    def get_all_scale_kde(self):
        self.kde_fn = []
        for ints in self.int_populations:
            self.kde_fn.append(self.get_scale_kde_fn(ints))
        

    def evaluate_populations(self, Xref, Yref):
        JSD = []
        if not hasattr(self, "kde_fn"):
            self.get_all_scale_kde()

        for kde in self.kde_fn:
            Ypop = kde(Xref)
            JSD.append(jensenshannon(Yref, Ypop))
        return np.array(JSD)

        
##########################################################
### Checking algorithm converges


### Run the scale generation procedure with different parameters,
### for the purpose of checking convergence
def convergence_check(cfn, params, Xref, Yref):
    n_step = [10, 20]
    beta_max = [1, 100]
    n_ints = 7
    results = []
    for ns in n_step:
        kwargs = {"n_step":ns, "beta_min": 0.001, "beta_arr_scale":"log"}
        for bm in beta_max:
            gen = ScaleGenerator(n_ints, bm, cfn, params, n_step=ns)
            gen.run()
            jsd = gen.evaluate_populations(Xref, Yref)
            results.append([gen.beta_arr, jsd])
    return results


### For each model, re-run the scale generation procedure many times,
### and save the data. This data can be used to check convergence.
def check_multiple_models():
    cfn = ['H_GFO_20', 'H_GP_1_20', "H_HP_3_1", "C_CI", "C_NI"]
    params = [{'I/A':'A', 'CF':'CF1'}] * 3 + [{'I/A':'A', 'RA':'rel'}, {'I/A':'A', 'W':20}]
    results = {st:{} for st in SCALE_TYPE}
    for i, st in enumerate(SCALE_TYPE):
        Xref, Yref = np.load(PATH_COMP.joinpath(f"ScaleDegreeDist/{st}_7_20.npy")).T
        for j in range(5):
            print(st, cfn[j])
            results[st][cfn[j]] = convergence_check(cfn[j], params[j], Xref, Yref)
    path = PATH_RES.joinpath("GenModel", "mc_convergence.pkl")
    pickle.dump(results, open(path, 'wb'))



##########################################################
### Generating populations for hypothesis testing


### Generates populations of scales at different values of beta
### for the input model and parameters
def get_int_population(i, cfn, params, beta, n=7, octave=False, bias_ints=True):
    # Need to initialize a random seed when using multiprocessing.Pool
    np.random.seed(int(str(time.time()).split('.')[1]))

    name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"
    path_pop = PATH_COMP.joinpath("GenPop", f"ints_{name}_{n}_{i:03d}{otxt}{btxt}.npy")
    if path_pop.exists():
        return
    print(name)

    gen = ScaleGenerator(n, beta, cfn, params, octave=octave, bias_ints=bias_ints)
    gen.run()
    ints = np.array(gen.int_populations[0])
    np.save(path_pop, ints)
    

### The selected models to be used in this work.
### See cost_functions.py to see what the names mean
def selected_models():
    cfn = ['H_GP_1_20', 'H_GFO_20', "H_HP_3_1", "H_HP_39_1", "C_CI", "C_NI", "C_NI"]
    params = [{'I/A':'A', 'CF':'CF1'}] * 4 + [{'I/A':'A', 'RA':'rel'},
              {'I/A':'A', 'W':20}, {'I/A':'I', 'W':20}]
    beta_list = np.logspace(-4, 2, 61)
    return cfn, params, beta_list


### Generate populations of scales for all models
def generate_populations(nrep=10, octave=False, bias_ints=True):
    cfn, params, beta_list = selected_models()
    n_ints = EM.N_INTS
    for i, (c, p) in enumerate(zip(cfn, params)):
        with Pool(N_PROC) as pool:
            inputs = [(i, c, p, b, n, octave, bias_ints) for i in range(nrep) for b in beta_list for n in n_ints]
            pool.starmap(get_int_population, inputs, 20)
        


##########################################################
### Evaluate generated populations

### Calculate all cost functions for generated populations
def get_cost(ints, cfn, cfn_params, octave=False, imax=1250, redo=False):
    modify_cost_fn = False
    params = {}
    for k, v in cfn_params.items():
        if k == 'I/A':
            if v == 'A':
                if octave:
                    ints = CF.get_all_ints(ints, 'circular')
                else:
                    ints = CF.get_all_ints(ints, 'first')

        elif k == 'CF':
            modify_cost_fn = True
            modify_fn = {'CF1': lambda x: -x,
                         'CF2': lambda x: (1 / (1 + x))}[v]
        else:
            params[k] = v

    if cfn[0] in 'HI':
        ints = np.round(ints.copy()).astype(int)
        ints[ints >= imax] = 0
        score = EM.get_model_score(cfn)
        cost = np.nanmean(score[ints], axis=1)

    elif cfn[:4] == 'C_CI':
        # We fix M = 1
        cost = CF.template_function_many(ints, [1])
        # Choosing rel vs abs
        cost = cost[0][0] if params['RA'] == 'rel' else cost[1][0]

    elif cfn[:4] == 'C_NI':
        cost = CF.number_unique_intervals_many(ints, **params)

    if modify_cost_fn:
        return modify_fn(cost)
    return cost



### Load generated populations, calculate all cost functions,
### and compare the scale degree distributions with empirical distributions
def evaluate_one(i, cfn, params, beta, n=7, octave=False, bias_ints=True, redo=False):
    name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"
    path_pop = PATH_COMP.joinpath("GenPop", f"ints_{name}_{n}_{i:03d}{otxt}{btxt}.npy")

    if not path_pop.exists():
        return
    print(name)

    ints = np.load(path_pop)

    path_cost = PATH_COMP.joinpath("GenPop", f"cost_{name}_{n}_{i:03d}{otxt}{btxt}.npy")
    if not path_cost.exists() or redo:
        cfn, params, beta_list = selected_models()
        cost = np.array([get_cost(ints, c, p, octave=octave) for c, p in zip(cfn, params)])
        np.save(path_cost, cost)

    path_jsd = PATH_COMP.joinpath("GenPop", f"jsd_{name}_{n}_{i:03d}{otxt}{btxt}.npy")
    if not path_jsd.exists() or redo:
        kde_fn = gaussian_kde(np.cumsum(ints, axis=1).ravel(), bw_method=0.05)
        JSD = []
        for i, st in enumerate(SCALE_TYPE):
            path_scaledegree = PATH_COMP.joinpath(f"ScaleDegreeDist/{st}_{n}_20.npy")
            if not path_scaledegree.exists():
                continue
            Xref, Yref = np.load(path_scaledegree).T
            Ypop = kde_fn(Xref)
            JSD.append(jensenshannon(Yref, Ypop))
        np.save(path_jsd, JSD)
    

### Evaluate all generated populations
def evaluate_populations(nrep=10, octave=False, bias_ints=True):
    cfn, params, beta_list = selected_models()
    n_ints = EM.N_INTS
    for i, (c, p) in enumerate(zip(cfn, params)):
        with Pool(N_PROC) as pool:
            inputs = [(i, c, p, b, n, octave, bias_ints) for i in range(nrep) for b in beta_list for n in n_ints]
            pool.starmap(evaluate_one, inputs, 20)
    

### Load all evaluation results: 
###     either cost functions, "res='cost'"
###     or JSD between scale degree distributions, "res='jsd'"
def load_results(i0=0, res='cost', octave=False, n_ints=7, bias_ints=True, nrep=10):
    cfn, params, beta_list = selected_models()
    cfn, params = cfn[i0], params[i0]
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"   # No bias defaults to the Melody model

    results = []
    for beta in beta_list:
        for j in range(nrep):
            name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
            path = PATH_COMP.joinpath("GenPop", f"{res}_{name}_{n_ints}_{j:03d}{otxt}{btxt}.npy")
            results.append(np.load(path))
    return np.array(results).reshape((len(beta_list), nrep, ) + results[0].shape)
                
    
### Caclculate correlations between cost functions in generated populations
def calculate_correlations(n_ints=7):
    path_corr = PATH_RES.joinpath("GenModel", f"cfn_corr_beta_{n_ints}.pkl")
    print(path_corr)
    if path_corr.exists():
        return pickle.load(open(path_corr, 'rb'))

    cfn, params, beta_list = selected_models()
    n = len(cfn)
    res_cost = np.array([load_results(i, 'cost') for i in range(n)])
    corr = []
    corr_names = []
    for j, res in enumerate(res_cost):
        print(j, cfn[j])
        for k in range(n):
            corr.append([utils.get_lr(res[i,:,j].ravel(), res[i,:,k].ravel()) for i in range(len(beta_list))])
            corr_names.append(f"{cfn[j]} :: {cfn[k]}")
    corr = np.array(corr).reshape(n, n, len(beta_list))
    out = {'beta':beta_list, 'names':corr_names, 'corr':corr}
    pickle.dump(out, open(path_corr, 'wb'))
    return out


### Plot step size distributions using kernel density estimate (KDE) functions
def get_kde_plot(cfn, params, beta, n=7, redo=False, nrep=10, xmax=1200, octave=False, bias_ints=False):
    name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"
    path_kde = PATH_RES.joinpath("GenPopStepSize", f"kde_{name}_{n}{otxt}{btxt}.npy")
    if path_kde.exists() and not redo:
        return np.load(path_kde)

    X = np.arange(xmax + 1)
    kde = []
    for i in range(nrep):
        path_pop = PATH_COMP.joinpath("GenPop", f"ints_{name}_{n}_{i:03d}{otxt}{btxt}.npy")
        kde_fn = gaussian_kde(np.load(path_pop).ravel(), bw_method=0.05)
        kde.append(kde_fn(X))
    kde = np.mean(kde, axis=0)
    np.save(path_kde, kde)
    return kde
    
    
def step_int_kde_plots(redo=False, nrep=10, octave=False):
    cfn, params, beta_list = selected_models()
    n_ints = np.arange(2, 10)
    inputs = [(c, p, b, n, redo, nrep, 1200, octave) for c, p in zip(cfn, params) for b in beta_list for n in n_ints]
    with Pool(N_PROC) as pool:
        return np.array(pool.starmap(get_kde_plot, inputs, 20)).reshape(len(cfn), len(beta_list), len(n_ints), -1)


### Get minimum / maximum step size per scale, for a generated population
def get_minmax(cfn, params, beta, n=7, redo=False, nrep=10, xmax=1200, octave=False, bias_ints=False):
    name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"
    path_minmax = PATH_RES.joinpath("GenPopStepSize", f"minmax_{name}_{n}{otxt}{btxt}.npy")
    if path_minmax.exists() and not redo:
        return np.load(path_minmax)

    X = np.arange(xmax + 1)
    minmax = []
    for i in range(nrep):
        path_pop = PATH_COMP.joinpath("GenPop", f"ints_{name}_{n}_{i:03d}{otxt}{btxt}.npy")
        ints = np.load(path_pop)
        minmax.append([np.min(ints, axis=1), np.max(ints, axis=1)])
    minmax = np.array(minmax).mean(axis=(0, 2))
    np.save(path_minmax, minmax)
    return minmax


### Get minimum / maximum step size per scale, for all generated populations
def step_int_minmax(redo=False, nrep=10, octave=False):
    cfn, params, beta_list = selected_models()
    n_ints = np.arange(2, 10)
    inputs = [(c, p, b, n, redo, nrep, 1200, octave) for c, p in zip(cfn, params) for b in beta_list for n in n_ints]
    with Pool(N_PROC) as pool:
        return np.array(pool.starmap(get_minmax, inputs, 20)).reshape(len(cfn), len(beta_list), len(n_ints), -1)


### Generate and evaluate scales for interferences models
def interference_models(nrep=10):
    best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open('../Figures/Data/si_fig3c_data_A_o20.pkl', 'rb'))
    params = CF.get_model_params()
    p = {'I/A':'A', 'CF':'CF1'}
    beta_list = np.logspace(-4, 2, 61)
    octave = [0, 0, 1]
    n = 7
    inputs = []
    for i, st in enumerate(SCALE_TYPE):
        for j, m in enumerate(HI_MODELS):
            name = "_".join(x for x in [m] + [str(p[k]) for p, k in zip(params[m], best_models[st][j][1][0][3:])])
            for b in beta_list:
                for k in range(nrep):
                    inputs.append((k, name, p, b, n, octave[i], True))
    with Pool(N_PROC) as pool:
        pool.starmap(get_int_population, inputs, 20)
        pool.starmap(evaluate_one, inputs, 20)


### Load interference model results ('cost', or 'jsd')
def load_results_interference(name, res='cost', octave=False, n_ints=7, bias_ints=True, nrep=10):
    otxt = "_octave" if octave else ""
    btxt = "" if bias_ints else "_nobias"
    results = []
    beta_list = selected_models()[2]
    for beta in beta_list:
        for j in range(nrep):
            name_beta = f"{name}_{np.log10(beta):05.2f}"
            path = PATH_COMP.joinpath("GenPop", f"{res}_{name_beta}_{n_ints}_{j:03d}{otxt}{btxt}.npy")
            results.append(np.load(path))
    return np.array(results).reshape((len(beta_list), nrep, ) + results[0].shape)


### Load all interference model results
def load_all_res_int(res='ints'):
    best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open('../Figures/Data/si_fig3c_data_A_o20.pkl', 'rb'))
    params = CF.get_model_params()
    octave = [0, 0, 1]
    n = 7
    result = {}
    for i, st in enumerate(SCALE_TYPE):
        result[st] = {}
        for j, m in enumerate(HI_MODELS):
            name = "_".join(x for x in [m] + [str(p[k]) for p, k in zip(params[m], best_models[st][j][1][0][3:])])
            result[st][m] = load_results_interference(f"{name}_A_CF1", res, octave=octave[i])
    return result
                
    



if __name__ == "__main__":

    # Run algorithm to generate data for checking convergence of alogrithm
    if 0:
        check_multiple_models()
    # Generate and evaluate all models
    if 1:
        for o in [0,1]:
            generate_populations(octave=o)
            evaluate_populations(octave=o)
            generate_populations(octave=o, bias_ints=False)
            evaluate_populations(octave=o, bias_ints=False)
        interference_models()



