"""
This module, "si_figs.py", contains code to reproduce supplementary figures.

Some figures (S5, S6, S25) require additional data to be downloaded,
from other research projects, which is all freely available online.
For more information:
    S5 :: see prod_variance.py
    S6 :: see sdt_fit.py
    S25 :: see line 1202 of this module

The rest of the figures are in principle reproducible,
however some of them require precomputation of cached data,
which is performed when running other code provided here, e.g., evaluate_model.py.

"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import brute, minimize
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist
from scipy.stats import norm, pearsonr, gaussian_kde, linregress, shapiro, entropy
import seaborn as sns

import cost_functions as CF
import evaluate_models as EM
import generate_scales as GS
import interval_constraints as IC
import interval_stats as IS
import paper_figs
from paper_figs import COL_ST
import scales_io
from scales_config import *
import scales_utils as utils
import weighted_sampling as WS


def load_real_costs(n_ints, octave=False):
    names0 = EM.all_model_names()
    txt = 'octave' if octave else "1250"
    costs = np.load(PATH_COMP.joinpath(f"CostFn/EmpiricalCost/{txt}_{n_ints}.npy"))
    idx = [np.where(names0 == name)[0][0] for i, name in enumerate(EM.model_name_examples())]
    return costs[idx], names0[idx]


#####################################
### SI Fig S2-S4
### Complexity model figures


def complexity_fig2(df):
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(4,6)
    ax = [fig.add_subplot(gs[:2,:3]), fig.add_subplot(gs[:2,3:])] + \
         [fig.add_subplot(gs[2:4,:4])]
    fig.subplots_adjust(wspace=0.8, hspace=1.00)
    fig2c(df, ax[:2])
    fig2fg(df, ax[2])

    fs = 12
    xi = [(-.15, 1.08), (-.06, 1.08), (-0.08, 0.98), (-0.08, 0.98), (-.10, 1.03), (-.20, 1.03), (-.20, 1.03), (-.15, 1.03)]
    for i, (a, b) in enumerate(zip(ax, 'ABCDEFGH')):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.text(*xi[i], b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si_complexity_fig2.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si_complexity_fig2.png"), bbox_inches='tight')


def fig2c(df, ax='', st='Vocal', n_rep=10000, scale_range=1700):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(8,4))

    df = df.loc[df.ScaleType==st]
    N = np.arange(2, 11)
    step = np.array([x for y in df.Ints for x in y])

    # Get Harmony predictions
    minmax = GS.step_int_minmax()

    fn_list = [np.min, np.max]
    pat = ['-o', '-s', ':^']
    lbls = ['Null', 'Empirical', 'Melody', 'Complexity']
    col = ['grey', 'k'] + sns.color_palette()[:3]
    ylbls = ['Mean Minimum Step,\n' + r'$\langle \min \{S \} \rangle$, (semitones)',
             'Mean Maximum Step,\n' + r'$\langle \max \{S \} \rangle$, (semitones)']
    for i, fn in enumerate(fn_list):
        Y1 = np.array([np.mean([fn(x) for x in df.loc[df.n_notes==n, 'Ints']]) for n in N])
        Y2 = np.array([IC.estimate_stat_limit(step, n - 1, n_rep, fn, scale_range) for n in N])
        if i == 0:
            Y3 = scale_range / (N * (N - 1))
        else:
            Y3 = np.load('../Precompute/expected_max.npy')[:len(N)] * scale_range
        for j, Y in enumerate([Y3, Y1, Y2]):
            ax[i].plot(N - 1, Y / 100, pat[j], ms=10, fillstyle='none', label=lbls[j], c=col[j])

        ax[i].plot(N[:8], minmax[3,50,:,i] / 100, ':<', fillstyle='none', ms=10, label='Harmony', c=col[3])
        ax[i].plot(N[:8], minmax[4,54,:,i] / 100, ':>', fillstyle='none', ms=10, label='Complexity', c=col[4])

        ax[i].set_xlabel(r"Number of step intervals, $N_I$")
        ax[i].set_ylabel(ylbls[i])
        if i == 1:
            handles, labels = ax[i].get_legend_handles_labels()
            handles = handles[:2] + [Patch(color='white')] + handles[2:]
            labels = labels[:2] + [' '] + labels[2:]
            ax[i].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.12), frameon=0, ncol=2)
    ax[0].set_yticks(range(5))
    ax[0].set_ylim(0, 4)
    ax[1].set_ylim(0, 9)


def fig2fg(df, ax='', st='Vocal', c0=207.4, octave=False):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    step = np.array([x for y in df.loc[(df.n_notes>1)&(df.ScaleType==st), "Ints"] for x in y])
    X = np.arange(0, 1201)
    Y0 = gaussian_kde(step)(X)
    ax.plot(X / 100, Y0, '-k')

    X = np.arange(1201) / 100
    beta_list = GS.selected_models()[2]
    i0_list = [2, 3]
    j0_list = [2, 4]
    k0 = [50, 54]
    n = 7
    lbls = ['Harmony', 'Complexity']

    col = sns.color_palette()
    for i, (i0, j0) in enumerate(zip(i0_list, j0_list)):
        kde = GS.step_int_kde_plots(octave=octave)[j0]
        ax.plot(X, kde[k0[i],3], c=col[i+1])

    X = np.arange(0, 1201)
    col = np.array(sns.color_palette('Paired'))[[1, 0, 0]]
    c_list = [c0, 138.6, 304.5]

    bounds = [(10, 200), (1, 100), (1, 500)]
    x0 = [50, 20, 100]
    a, b, c = minimize(IC.fit_fn, x0, args=(X, Y0, IC.interval_probability_function), bounds=bounds).x
    print(a, b, c)
    Y = IC.interval_probability_function(X, a, b, c)
    ax.plot(X / 100, Y / np.trapz(Y, X), ':', c=col[0], label='Melody - no constraints')

    Y1 = []
    for i, c in enumerate(c_list):
        a, b = minimize(IC.fit_constraint_c, x0[:2], args=(X, Y0, c, 'cdf'), bounds=bounds[:2]).x
        print(a, b, c)
        y = IC.interval_probability_function(X, a, b, c)
        Y1.append(y / np.trapz(y, X))
    ax.plot(X / 100, Y1[0], color=col[0])
    Ylo = np.min(Y1[1:], axis=0)
    Yhi = np.max(Y1[1:], axis=0)
    ax.fill_between(X / 100, Ylo, Yhi, color=col[1])

    ax.set_xlabel(r"Step Interval, $I_S$ (semitones)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 10)

    handles = [Line2D([], [], ls='-', c='k'), Line2D([], [], ls=':', c=col[0])] + \
              [Line2D([], [], ls='-', c=c) for c in sns.color_palette()[:3]]
    labels = ['Emprirical', 'Melody - unconstrained', 'Melody - constrained'] + lbls
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.80, 1.0), frameon=False, ncol=1)



def complexity_fig3(df, redo=False, nsamp_max="o20", modeA='resample', modeC='A'):
    fig = plt.figure(figsize=(12,12))

    gs = GridSpec(5,6,height_ratios=[1, .4, 1.2, .5, .8])
    ax = [fig.add_subplot(gs[0,i:i+2]) for i in [0,2,4]] + \
         [fig.add_subplot(gs[2,i:i+3]) for i in [0,3]] + \
         [fig.add_subplot(gs[4,i:i+3]) for i in [0,3]]

    fig.subplots_adjust(wspace=1.0, hspace=0)

    fig3b(nsamp_max if isinstance(nsamp_max, int) else int(nsamp_max[1:]), ax[:3], redo)
    fig3d(redo, ax[3:5], modeC, nsamp_max)
    fig3e(ax=ax[5:])

    fs = 12
    for i, a in enumerate(ax):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for i, b in zip([0, 3, 5], 'ABCD'):
        ax[i].text(-.10, 1.07, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si_complexity_fig3.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si_complexity_fig3.png"), bbox_inches='tight')


def get_histograms(data, bins):
    X = bins[:-1] + np.diff(bins[:2]) / 2
    hist = np.array([np.histogram(d, bins=bins)[0] / len(d) for d in data])
    return X, hist


def fig3b(nsamp=20, ax='', redo=False):
    real_cost, ran_cost = IS.boot_scale_costs(nsamp=nsamp, redo=redo)
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,3,figsize=(9,4))

    yoffset = 0.5
    xlim = (0.5, 8.6)
    bins = np.arange(*xlim)
    m = MAIN_MODELS[3]
    al = 0.7
    fs = 8

    for j, st in enumerate(SCALE_TYPE):
        width = np.diff(bins[:2]) / 3
        X1, hist_real = get_histograms(real_cost[st][:,3], bins)
        X2, hist_ran = get_histograms(ran_cost[st][:,3], bins)

        yoff = yoffset * (2 - j)
        xoff = width / 2
        ax[j].bar(X1 - xoff, np.mean(hist_real, axis=0), width, bottom=yoff, color=COL_ST[st], label=st, ec='k', lw=0.4, alpha=0.5)
        ax[j].bar(X2 + xoff, np.mean(hist_ran, axis=0), width, bottom=yoff, color=sns.color_palette()[0], ec='k', lw=0.4, alpha=0.5)
        ax[j].set_xlim(xlim)
        ax[j].set_yticks([])
        ax[j].set_ylabel("Probability")
        ax[j].set_xlabel("Unique step intervals\n$\mathcal{A}_S$")
        ax[j].set_xticks(range(1, 9))

        handles = [Patch(ec='k', fc=COL_ST[st], alpha=al), Patch(ec='k', fc=sns.color_palette()[0], alpha=al)]
        lbls = [st, 'Melody']
        ax[j].legend(handles, lbls, ncol=1, frameon=False, fontsize=fs,
                       loc='upper left', bbox_to_anchor=(0.50, 1.15), handlelength=.9)


def fig3d(redo=False, ax='', mode='A', nsamp_max="o20"):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(8,4))

    df = scales_io.load_all_data()
    df = df.loc[df.n_ints.isin(EM.N_INTS)]
    
    path = PATH_FIG_DATA.joinpath(f"scale_prob.csv")
    if path.exists() and not redo:
        dfr = pd.read_csv(path)
    else:
        model_list, st_list, prob_list, w_list = [], [], [], []
        for j, st in enumerate(SCALE_TYPE):
            ist, insamp = EM.get_weight_index(st, nsamp_max)
            otxt = 'octave' if st == 'Theory' else '1250'
            weightsO = WS.get_weights_by_region_overall(df)[ist, insamp]

            for model in ['H_GFO', 'H_HP', 'C_NI']:
                sprob_n = {n: np.load(f'../Precompute/CostFn/ScaleProb/{model}_{otxt}_{n}.npy') for n in EM.N_INTS}
                sprob0 = np.zeros(sprob_n[5].shape[:-1] + (len(df), ), float)
                for n in EM.N_INTS:
                    if len(sprob0.shape) == 6:
                        sprob0[:,:,:,:,:,df.n_ints==n] = sprob_n[n]
                    elif len(sprob0.shape) == 5:
                        sprob0[:,:,:,:,df.n_ints==n] = sprob_n[n]
                    elif len(sprob0.shape) == 4:
                        sprob0[:,:,:,df.n_ints==n] = sprob_n[n]
                sprob0[~np.isfinite(sprob0)] = np.nan

                mprob0 = np.average(np.log(sprob0), weights=weightsO, axis=-1)

                idx = np.unravel_index(np.nanargmax(mprob0), mprob0.shape)
                idx_st = df.ScaleType==st
                prob_list.extend(list(sprob0[idx][idx_st]))
                model_list.extend([model] * np.sum(idx_st))
                st_list.extend([st] * np.sum(idx_st))
                w_list.extend(list(weightsO[idx_st]))
        dfr = pd.DataFrame(data={'model':model_list, 'ScaleType':st_list, 'prob':prob_list, 'weight':w_list})
        dfr['logprob'] = np.log(dfr['prob'])
        dfr.to_csv(path, index=0)

    dfr = dfr.loc[dfr.model.isin(['H_HP', 'C_NI'])]

    sns.boxenplot(x='model', y='logprob', hue='ScaleType', data=dfr, showfliers=False, ax=ax[1], palette=COL_ST)

    xlim = ax[1].get_xlim()
    ax[1].plot(xlim, [0,0], ':k')
    ax[1].set_xlim(xlim)

    X = np.arange(2)
    al = 0.7
    fs = 12
    width = 0.25
    pat = ['-', '--', ':']
    for i, st in enumerate(SCALE_TYPE):
        # Plot LLR
        xoff = (i - 1) * width
        Y = []
        for model in ['H_HP', 'C_NI']:
            x, y = dfr.loc[(dfr.model==model)&(dfr.ScaleType==st), ['logprob', 'weight']].values.T
            Y.append(np.average(x, weights=y))
            null_2sigma = np.array([2 / (np.sum(y))**0.5] * X.size)
        Y = np.array(Y)
        print(np.exp(Y))
        ax[0].bar(X + xoff, Y, width, color=COL_ST[st], alpha=al, ec='k', lw=0.5)

        idx1 = null_2sigma < Y
        idx2 = (null_2sigma * 1.5) < Y
        if st == 'Theory':
            ax[0].plot(X[idx1] + xoff, Y[idx1] + 0.4, '*k')
            ax[0].plot(X[idx2] + xoff, Y[idx2] + 0.8, '*k')
        else:
            ax[0].plot(X[idx1] + xoff, Y[idx1] + 0.2, '*k')
            ax[0].plot(X[idx2] + xoff, Y[idx2] + 0.5, '*k')


    xlbls = ['Harmony +\nMelody', 'Complexity +\nMelody']
    for i, a in enumerate(ax):
        a.set_xticks(X)
        a.set_xticklabels(xlbls)
        a.set_xlabel('')
    ax[0].set_ylabel("Log-likelihood ratio per scale")
    ax[1].set_ylabel(r"Log-likelihood ratio")
    ax[1].legend(frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.10))
    



def map_sig_col(sig_arr, colors):
    col_arr = np.zeros(sig_arr.shape + (3,), float)

    col_arr[np.isnan(sig_arr)] = colors[0]
    col_arr[sig_arr < 1] = colors[1]
    col_arr[(sig_arr >= 1) & (sig_arr < 1.5)] = colors[2]
    col_arr[(sig_arr >= 1.5) & (sig_arr < 3)] = colors[3]
    col_arr[sig_arr >= 3.0] = colors[4]

    return col_arr


def fig3e(redo=False, ax='', mode='A', nsamp_max="o20"):
    if isinstance(ax, str):
        fig, ax = plt.subplots(2,1,figsize=(8,6))

    # Pre-load data / save for future use
    path_data = PATH_FIG_DATA.joinpath(f"fig3c_data_{mode}_{nsamp_max}.pkl")
    if path_data.exists() and not redo:
        best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open(path_data, 'rb'))
    else:
        df = scales_io.load_all_data()
        weights = WS.get_weights_by_region(df)
        weightsR = WS.get_weights_within_regions(df)
        weightsO = WS.get_weights_by_region_overall(df)
        regions = np.array(sorted(df.Region.unique()))
        best_models = {}
        Yn, reg0, Yreg, wsum_list = [{st:[] for st in SCALE_TYPE} for _ in range(4)]
        for i, st in enumerate(SCALE_TYPE):
            # Find the best models, and save both LLR and model indices
            best_models[st] = [EM.best_model_across_n(m, st, nsamp_max, mode) for m in MAIN_MODELS]
            
            # Best-fitting model indices
            idx0 = {m: best_models[st][j][1][0] for j, m in enumerate(MAIN_MODELS)}

            # Get LLR per N (model indices are fixed)
            for j, m in enumerate(MAIN_MODELS):
                # Using the best-fitting model indices, get LLR for each N
                Yn[st].append(EM.best_model_across_n(m, st, nsamp_max, 'D'+mode, sep_n=1, idx0=idx0[m])[2])

            # Get LLR per Region (model indices are free)
            for j, reg in enumerate(regions):
                ist, insamp = EM.get_weight_index(st, reg)
                wsum = np.sum([weightsR[n][ist,insamp].sum() for n in EM.N_INTS])
                if wsum >= 10:
                    reg0[st].append(reg)
                    wsum_list[st].append(wsum)
                    Yreg[st].append([EM.best_model_across_n(m, st, reg, mode)[0] for m in MAIN_MODELS])
            Yreg[st] = np.array(Yreg[st])
        data = [best_models, Yn, reg0, Yreg, wsum_list]
        pickle.dump(data, open(path_data, 'wb'))

    sig_arr = np.zeros((3, 11, 4), float) * np.nan
    reg_key = {r: i for i, r in enumerate(REGIONS)}

    for i, st in enumerate(SCALE_TYPE):
        # Plot LLR vs Region
        n_reg = len(reg0[st])
        width = 0.8 / n_reg
        for j, (y, r) in enumerate(zip(Yreg[st], reg0[st])):
            if mode == 'A':
                null_2sigma = 2 / np.array(wsum_list[st][j])**0.5
            elif mode == 'B':
                # Need to account for fitting separate values of beta
                ist, insamp = EM.get_weight_index(st, r)
                n_beta_vals = np.sum([np.any(weightsR[n][ist,insamp] > 0) for n in EM.N_INTS])
                null_2sigma = 2 * (n_beta_vals / wsum_list[st][j])**0.5
            print(st, r, y)
            sig_arr[i,reg_key[r]] = y / null_2sigma

    colors = [(1,1,1), (0.8, 0.8, 0.8)] + sns.color_palette('dark:salmon_r', n_colors=4)
    hatch = ['////'] + [''] * 4
    col_arr = map_sig_col(sig_arr, colors)

    fs = 8
    ms = 8
    dx = 0.15
    al = 0.85
    ttls = ['Harmony + Melody', 'Complexity + Melody']
    for i, j in enumerate([2,3]):
        im = ax[i].imshow(col_arr[:,:,j], alpha=al, aspect='auto')
        ax[i].set_yticks(range(3))
        ax[i].set_yticklabels(SCALE_TYPE)
        ax[i].set_xticks(range(len(REGIONS)))
        ax[i].set_ylim(2.49, -0.49)
        ax[i].set_title(ttls[i], loc='left')

        for i0, j0 in zip(*np.where(np.isnan(sig_arr[:,:,j]))):
            ax[i].add_patch(Rectangle([j0 - 0.5, i0 - 0.5], 1, 1, 
                            hatch='//', fill=False, snap=False, linewidth=0))
    reg_full = [s.replace(' ', '\n') for s in REG_FULL]
    ax[0].set_xticklabels(reg_full, rotation=90)
    ax[1].set_xticklabels(reg_full, rotation=90)

    handles = [Patch(facecolor=c, edgecolor='k', hatch=h, alpha=al) for c, h in zip(colors, hatch)]
    lbls = ['Too few data', r'$p > 0.05$', r'$p < 0.05$', r'$p < 0.005$', r'$p < 10^{-8}$']
    ax[0].legend(handles, lbls, loc='upper left', bbox_to_anchor=(0.40, 1.25),
                 frameon=False, handlelength=0.9, ncol=3, columnspacing=1.0, fontsize=fs)

    return sig_arr
    

def complexity_fig4(df):
    fig = plt.figure(figsize=(10,7))

    gs = GridSpec(4,2,height_ratios=[.2,1,1,1])
    ax = np.array([[fig.add_subplot(gs[i,j]) for i in [1,2,3]] for j in [0,1]])
    cax = [fig.add_subplot(gs[0,i]) for i in [0,1]]
    fig.subplots_adjust(wspace=0.4, hspace=0.7)

    epiphenomena(df, ax=ax, cax=cax)

    fs = 14
    for i, a in enumerate(ax.ravel()):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for i, b in zip([0, 1], 'AB'):
        ax[i,0].text(-.1, 1.15, b, transform=ax[i,0].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si_complexity_fig4.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si_complexity_fig4.png"), bbox_inches='tight')


def epiphenomena(df, cost=None, ax='', cax=''):
    n_ints = np.arange(5, 8)
    if isinstance(cost, type(None)):
        path = PATH_FIG_DATA.joinpath("gen_pop_cost.npy")
        if path.exists():
            cost = np.load(path)
        else:
            cost = np.array([[[GS.load_results(i, 'cost', o, n) for i in range(6)] for o in [0,1]] for n in n_ints])
            np.save(path, cost)

    cost = cost[3:]
    
    if isinstance(ax, str):
        fig, ax = plt.subplots(2,3,figsize=(14,8))

    yt = [[5,10], [5,10], [10, 15]]
    ms = 6
    for i, n in enumerate(n_ints):
        for j, (o, p) in enumerate(zip([0,1], '-:')):
            c1 = -cost[i,o,1,:,:,1].mean(axis=(1,2))
            c2 = cost[i,o,1,:,:,5].mean(axis=(1,2))
            c3 = -cost[i,o,5,:,:,1].mean(axis=(1,2))
            c4 = cost[i,o,5,:,:,5].mean(axis=(1,2))
            paper_figs.multicolored_line(c1, c2, ax[o,i], 'flare')
            paper_figs.multicolored_line(c3, c4, ax[o,i], "crest")
            ax[o,i].text(0.7, 0.8, f"$N_I$ = {n}", transform=ax[o,i].transAxes)
            ax[o,i].set_xlabel(r"$\langle H_{OF} \rangle$")
            ax[o,i].set_ylabel(r"$\mathcal{A}_{I}$")
            ax[o,i].set_xticks([0, 1])
            ax[o,i].set_yticks(yt[i])
            ax[o,i].plot([c1[0]], [c2[0]], 'ok', ms=ms, fillstyle='full', mec='k', label='Melody')

        for st in SCALE_TYPE:
            c0, n0 = load_real_costs(n, 0)
            c1, n1 = load_real_costs(n, 1)

            idx = df.loc[df.n_ints==n, 'ScaleType'] == st
            ax[0,i].plot([-c0[1,idx].mean()], [c0[5,idx].mean()], 'o', c=COL_ST[st], ms=ms, fillstyle='full', mec='k', label=st)

            idx = idx & (np.abs(df.loc[df.n_ints==n, 'scale_range'] - 1200) <= 20)
            ax[1,i].plot([-c1[1,idx].mean()], [c1[5,idx].mean()], 'o', c=COL_ST[st], ms=ms, fillstyle='full', mec='k')
    fs = 10
    ax[0,0].set_yticks([5,10])
    ax[0,0].set_title("Non-Octave Scales", fontsize=fs)
    ax[1,0].set_title("Octave Scales", fontsize=fs)
    ax[0,0].legend(bbox_to_anchor=(1.30, 1.8), frameon=False, handletextpad=0.2, ncol=1)

    if not isinstance(cax, str):
        X = np.linspace(0,1,10).reshape(1, -1)
        im = cax[0].imshow(X, cmap=sns.color_palette('flare', as_cmap=1))
        im = cax[1].imshow(X, cmap=sns.color_palette("crest", as_cmap=1))
        for a in cax:
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlabel(r"Bias strength, $\beta$")
        cax[0].set_title("Harmony")
        cax[1].set_title("Complexity")
    


################################################################
### SI Fig S29
### Melodic interval distributions / cross-cultural variance

def sampling_gini_1(df):
    nsamp_max = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(7,4))
    for i, st in enumerate(SCALE_TYPE):
        count = df.loc[df.ScaleType==st].Region.value_counts()
        G = []
        for n in nsamp_max:
            prob = np.minimum(count, n)
            prob = prob / prob.sum()
            G.append(utils.gini(prob))
        ax.plot(nsamp_max, G, c=COL_ST[st], label=st)
    ax.plot([20, 20], [0, 0.55], ':k')
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Max. number of samples per region")
    ax.set_ylabel("Gini coefficient")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=False)
    
    fig.savefig(PATH_FIG.joinpath(f"si_gini.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_gini.png"), bbox_inches='tight')


################################################################
### SI Fig S9-S13
### Step size predictions of different models

### Model is chosen by i0, integers from 0 to 5
### See generate_scales.selected_models to see the corresponding models
def step_size_predictions(df, i0=0):
    kde = GS.step_int_kde_plots()[i0]
    fig, ax = plt.subplots(3,2,figsize=(9,7))
    ax = ax.reshape(ax.size)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    n_ints = np.arange(2, 8)
    idx = [[30, 45, 50, 55, 60]]*4 + [[30, 45, 50, 55, 60]] + [[30, 40, 45, 50, 55]] * 2
    X = np.arange(1201) / 100
    col = sns.color_palette("flare", n_colors=5)
    beta_list = GS.selected_models()[2]
    for i, n in enumerate(n_ints):
        for k, j in enumerate(idx[i0]):
            ax[i].plot(X, kde[j,i], c=col[k], label=f"$\\beta$ = {beta_list[j]:4.1f}")
        ax[i].set_ylim(0, 0.007)
        ax[i].set_xlim(0, 8)
        ax[i].set_xlabel("Step size (semitones)")
        ax[i].set_ylabel("Density")
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_title(f"N = {n}")

        Y0 = scales_io.load_int_dist_n(df, n)
        ax[i].plot(X, Y0, ':k')
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.05, 1.0), frameon=False, ncol=1)

    fig.savefig(PATH_FIG.joinpath(f"si_step_size_{i0}.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_step_size_{i0}.png"), bbox_inches='tight')


################################################################
### SI Fig S7
### Empirical step size distributions, broken down by region, scale type, and measurement type

def step_size_distributions_full(df, ax='', st=''):
    fig, ax = plt.subplots(2,1,figsize=(10,8))
    df = utils.rename_regions(df)
    dfi = IC.get_interval_df(df)
    int_median = dfi.groupby('Region')['Int'].median().sort_values()
    order = dfi.groupby('Region')['Int'].mean().sort_values().index
#   sns.boxenplot(x='Region', y='Int', data=dfi, ax=ax[0], showfliers=False, order=order, hue='ScaleType')
    sns.boxenplot(x='Region', y='Int', data=dfi, ax=ax[0], showfliers=True, order=order, hue='ScaleType')
#   sns.violinplot(x='Region', y='Int', data=dfi, ax=ax[0], order=order, hue='ScaleType')
#   sns.stripplot(x='Region', y='Int', data=dfi, ax=ax[0], order=order, hue='ScaleType', dodge=True)
    sns.boxenplot(x='Region', y='Int', data=dfi, ax=ax[1], showfliers=True, order=order, hue='Method', palette='dark')

    fs = 14
    for i, (a, b) in enumerate(zip(ax, 'ABCDEFGH')):
        a.set_ylim(0, 5)
        a.set_ylabel("Step Interval (semitones)")
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.legend(loc='upper left', frameon=0)
        a.text(-.05, 1.05, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si_step_size_dist.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_step_size_dist.png"), bbox_inches='tight')


################################################################
### SI Fig S8
### Recreation of main fig 3cd, but with Instrumental and Theory scales instead of Vocal

def fig2c(df, n_rep=10000):
    fig, ax = plt.subplots(2,2,figsize=(9,8))
    fig.subplots_adjust(hspace=0.6)

    scale_range = [1700, 1200]
    octave = [0, 1]
    N = np.arange(3, 11)
    ylbls = ['Mean Minimum Step,\n' + r'$\langle \min \{S \} \rangle$, (semitones)',
             'Mean Maximum Step,\n' + r'$\langle \max \{S \} \rangle$, (semitones)']
    pat = ['-o', '-s', ':^']
    lbls = ['Null', 'Empirical', 'Melody']
    col = ['grey', 'k'] + sns.color_palette()[:3]
    scale_types = ['Instrumental', 'Theory']
    for i0, st in enumerate(scale_types):
        df0 = df.loc[df.ScaleType==st]
        step = np.array([x for y in df.Ints for x in y])
        minmax = GS.step_int_minmax(octave=i0)

        fn_list = [np.min, np.max]
        for i, fn in enumerate(fn_list):
            # Empirical distribution
            Y1 = np.array([np.mean([fn(x) for x in df0.loc[df0.n_notes==n, 'Ints']]) for n in N])

            # IS-MC model
            Y2 = np.array([IC.estimate_stat_limit(step, n - 1, n_rep, fn, scale_range[i0], octave[i0]) for n in N])
            if i == 0:
                Y3 = scale_range[i0] / (N * (N - 1))
            else:
                Y3 = np.load('../Precompute/expected_max.npy')[:len(N)] * scale_range[i0]
            for j, Y in enumerate([Y3, Y1, Y2]):
                ax[i0,i].plot(N - 1, Y / 100, pat[j], ms=10, fillstyle='none', label=lbls[j], c=col[j])
            ax[i0,i].plot(N[:8], minmax[3,50,:,i] / 100, ':<', fillstyle='none', ms=10, label='Harmony', c=col[3])
            ax[i0,i].plot(N[:8], minmax[4,54,:,i] / 100, ':>', fillstyle='none', ms=10, label='Complexity', c=col[4])

            handles, labels = ax[i0,i].get_legend_handles_labels()
            handles = handles[:2] + [Patch(color='white')] + handles[2:]
            labels = labels[:2] + [' '] + labels[2:]
            ax[i0,i].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.20), frameon=0, ncol=2)
            ax[i0,i].set_xlabel(r"Number of intervals, $N_I$")
            ax[i0,i].set_ylabel(ylbls[i])
            ax[i0,i].spines['top'].set_visible(False)
            ax[i0,i].spines['right'].set_visible(False)
        ax[i0,0].set_title(st, loc='left')

    fs = 12
    for i, (a, b) in enumerate(zip(ax.ravel(), 'ABCD')):
        a.text(-.10, 1.05, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"step_min_max_IT.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"step_min_max_IT.png"), bbox_inches='tight')


################################################################
### SI Fig S14
### Direction and significance of tests for non-randomness of interval counts across regions

def plot_scale_degree_sig_region(df):
    fig, ax = plt.subplots(3,1,figsize=(14,8))
    fig.subplots_adjust(hspace=0.4)

    ax2 = np.array([a.twinx() for a in ax])
    X = np.arange(20, 1800, 20)
    prob = IS.get_sig_int_region(df)
    for i, (st, di) in enumerate(prob.items()):
        Y1 = np.mean([1 - v[1] for v in di.values()], axis=0)
        Y2 = np.mean([v[0] < 0.05 for v in di.values()], axis=0)
        idx = (X<1200) if st == 'Theory' else np.arange(X.size)

        ax[i].plot(X[idx] / 100, Y1[idx], '-o', c=sns.color_palette()[0], fillstyle='none', label="Fraction frequent")
        ax2[i].plot(X[idx] / 100, Y2[idx], '-o', c=sns.color_palette()[1], fillstyle='none', label="Fraction significant")
        ax[i].set_title(st, loc='left')
        ax[i].set_xlim(0, 17.5)
        ax[i].set_ylabel("Fraction frequent")
        ax2[i].set_ylabel("Fraction significant", rotation=270, labelpad=14)
        ax[i].legend(frameon=False, bbox_to_anchor=(0.4, 1.2))
        ax2[i].legend(frameon=False, bbox_to_anchor=(0.6, 1.2))

        ylim = ax[i].get_ylim()
        for x in range(1, 18):
            ax[i].plot([x,x], ylim, ':k', alpha=0.3)
        ax[i].set_ylim(ylim)
    ax[2].set_xlabel("Scale Degree (semitones)")

    fs = 12
    for i, (a, b) in enumerate(zip(ax.ravel(), 'ABCD')):
        a.text(-.05, 1.05, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"scale_degree_sig_reg.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"scale_degree_sig_reg.png"), bbox_inches='tight')


################################################################
### SI Fig S15
### Model comparison as a function of N

def llr_vs_n(mode='A', nsamp_max="o20"):
    fig, ax = plt.subplots(3,1,figsize=(6,6))
    fig.subplots_adjust(hspace=0.4)
    path_data = PATH_FIG_DATA.joinpath(f"fig3c_data_{mode}_{nsamp_max}.pkl")
    best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open(path_data, 'rb'))
    fs = 12
    n_ints = np.arange(2, 10)

    lbls = ['Harmony (GP)', 'Harmony (OF)', 'Harmony (HP)', 'Complexity']
    for i, st in enumerate(SCALE_TYPE):
        for j, (y, m) in enumerate(zip(Yn[st], MODEL_NAMES)):
            ax[i].plot(n_ints, y[:len(n_ints)], '-o', label=lbls[j])
        ax[i].set_title(st, loc='left', fontsize=fs)
        ax[i].set_xlabel(r"$N_I$")
        ax[i].set_ylabel(r"Log-likelihood Ratio")
        ax[i].plot([1.8, 9.2], [0,0], ':k')
        ax[i].set_title(st, loc='left', fontsize=fs)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    ax[0].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=fs-4,
                 handlelength=0.8, columnspacing=0.8, handletextpad=0.4)

    fig.savefig(PATH_FIG.joinpath(f"si_llr_vs_n.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_llr_vs_n.png"), bbox_inches='tight')


################################################################
### SI Fig S16
### HP Model performance as a function of its parameters (number of harmonics and roll-off)

def hp_params(ax='', nsamp_max="o20"):
    if isinstance(ax, str):
        fig, ax = plt.subplots(2,3,figsize=(12,14))
    n_partial = list(range(3, 41))
    rho = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20] 

    for i, st in enumerate(SCALE_TYPE):
        for j in [0,1]:
            mprob = np.nanmax(EM.best_model_across_n("H_HP", st, nsamp_max, 'A', po_only=True), axis=0)[1-j,0]
            im = ax[j,i].imshow(mprob, aspect='auto', cmap=sns.color_palette('flare', as_cmap=True))
            cbar = fig.colorbar(im, ax=ax[j,i], location='top')
            cbar.set_label('Log-likelihood Ratio')

            ax[j,i].invert_yaxis()
            ax[j,i].set_xticks(np.arange(len(rho)))
            ax[j,i].set_xticklabels(rho)
            ax[j,i].set_yticks(np.arange(len(n_partial)))
            ax[j,i].set_yticklabels(n_partial)
            ax[j,i].set_xlabel(r"Harmonic decay rate, $\rho$")
            ax[j,i].set_ylabel(r"Number of partials, $n$")
            ax[j,i].set_title(st, loc='left', pad=50)

    fs = 14
    for i, (a, b) in enumerate(zip(ax[:,0], 'AB')):
        a.text(-.15, 1.08, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si_hp_params.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_hp_params.png"), bbox_inches='tight')


################################################################
### SI Fig S28
### Sensitivity of various models to their parameters

def param_sens(nsamp_max='o20'):
    fig, ax = plt.subplots(2,3,figsize=(12,8))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    xlbl = ['Gaussian kernel width, w (cents)', 'Window size, w (cents)', 'Max. interval category std. dev., w (cents)']
    ttls = ["Harmony (OF)", "Harmony (GP)", 'Complexity']
    fix_vals = [1, 1, 0]

    for k, m in enumerate(['H_GFO', 'H_GP', 'C_NI']):
        for i, st in enumerate(SCALE_TYPE):
            for j in [0,1]:
                mprob = np.nanmax(EM.best_model_across_n(m, st, nsamp_max, 'A', po_only=True, fix_vals=fix_vals[k]), axis=0).reshape(2,-1)[1-j]
                ax[j,k].plot(CF.W_ARR, mprob, label=st, c=COL_ST[st])
                ax[j,k].set_xlabel(xlbl[k])
                ax[j,k].set_ylabel("Log-likelihood Ratio")
                ax[j,k].set_title(ttls[k], loc='left')
                ax[j,k].legend(loc='best', frameon=False)
        ax[0,k].set_yscale('log')
    ax[1,2].set_yscale('log')

    fs = 14
    for i, (a, b) in enumerate(zip(ax[:,0], 'AB')):
        a.text(-.15, 1.08, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si_param_sens.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_param_sens.png"), bbox_inches='tight')


################################################################
### SI Fig S17
### Fitting generated populations to empirical populations of scales

def jsd_vs_beta(n_ints=7):
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    cfn, params, beta_list = GS.selected_models()
    lbls = ['OF', 'GP', r'HP$^{\textrm{A}}$', r'HP$^{\textrm{B}}$']
    octave = [0, 0, 1]
    for j, st in enumerate(SCALE_TYPE):
        jsd = np.array([GS.load_results(i, 'jsd', octave[j], n_ints) for i in range(4)])
        for i, res in enumerate(jsd):
            rm = res[:,:,j].mean(axis=1)
            ax[j].plot(beta_list, rm, label=lbls[i])
        ax[j].legend(loc='best', frameon=False)
        ax[j].set_xscale('log')
        ax[j].set_xlabel(r"$\beta$")
        ax[j].set_ylabel("Jensen-Shannon Distance")
        ax[j].set_xlim(0.1, 100)
        ax[j].set_ylim(0, 0.5)
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['right'].set_visible(False)
        ax[j].set_title(st, loc='left')

    fig.savefig(PATH_FIG.joinpath(f"si_jsd_vs_beta.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_jsd_vs_beta.png"), bbox_inches='tight')


################################################################
### SI Fig S18-S21
### Scale degree distributions for generated populations of scales

def generated_scale_deg_dist(ax='', n_ints=7):
    if isinstance(ax, str):
        fig, ax = plt.subplots(3,1,figsize=(11,7))
        fig.subplots_adjust(hspace=0.4)

    jsd = np.array([GS.load_results(i, 'jsd', 0, n_ints) for i in range(4)]).mean(axis=2)
    jsd_octave = np.array([GS.load_results(i, 'jsd', 1, n_ints) for i in range(4)]).mean(axis=2)
    octave = [0,0,1]
    xmax = [15, 15, 13]
    lbls = ['OF', 'GP', r'HP$^{\textrm{A}}$', r'HP$^{\textrm{B}}$']
    for i, st in enumerate(SCALE_TYPE):
        for k, j in enumerate([1,0,2,3]):
            if st == 'Theory':
                imin = np.argmin(jsd_octave[j,:,i])
                Xref, Yref, Ypop = paper_figs.load_gen_pop_dist(i0=j, j0=imin, st=st, octave=1, n_ints=n_ints)
            else:
                imin = np.argmin(jsd[j,:,i])
                Xref, Yref, Ypop = paper_figs.load_gen_pop_dist(i0=j, j0=imin, st=st, octave=0, n_ints=n_ints)
            print(st, j, round(pearsonr(Ypop, Yref)[0], 2))
            ax[i].plot(Xref/100, Ypop, label=lbls[k])
        ax[i].plot(Xref/100, Yref, '-k', label="Data")
        ax[i].set_xlabel("Scale Degree (semitones)")
        ax[i].set_ylabel("Density")
        ax[i].set_yticks([])
        ax[i].set_title(st, loc='left', fontsize=10)
        ax[i].set_xlim(0,16)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        for x in range(1, xmax[i]):
            ylim = ax[i].get_ylim()
            ax[i].plot([x,x], ylim, ':', c='grey', alpha=0.5)
            ax[i].set_ylim(ylim)
#   ax[2].set_xlim(0,12.5)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.6, 1.5), ncol=3, frameon=0,
                       handlelength=1.5, columnspacing=1.0, handletextpad=0.4)

    fig.savefig(PATH_FIG.joinpath(f"si_gen_scale_deg_{n_ints}.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_gen_scale_deg_{n_ints}.png"), bbox_inches='tight')


################################################################
### SI Fig S30
### Correlations between harmonicity costs and complexity costs in empirical populations of scales

def HI_IT_cost_corr():
    fig, ax = plt.subplots(2,3,figsize=(12,7))
    fig.subplots_adjust(hspace=0.5)
    ax = ax.reshape(ax.size)
    n_ints = np.arange(3, 9)
    for i, n in enumerate(n_ints):
        X, Y = load_real_costs(n)[0][[1,5]]
        sns.regplot(x=-X, y=Y, ax=ax[i])
        ax[i].set_title(f"$N_I$ = {n}")
        ax[i].set_xlabel(r"$\langle H_{OF} \rangle$")
        ax[i].set_ylabel(r"$\mathcal{A}_{I}$")

        r, p = utils.get_lr(-X, Y, p=1)
        txt = f"$r$ = {r:5.2f} \n $p$ = {p:.1g}"
        ax[i].text(0.8, 1.0, txt, transform=ax[i].transAxes)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath(f"si_HI_IT_cost_corr.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_HI_IT_cost_corr.png"), bbox_inches='tight')
    

################################################################
### SI Fig S22
### Number of scale degrees per scale, shown per region

def scale_size_per_reg(df):
    fig, ax = plt.subplots(figsize=(8,4))
    order = df.loc[:, ['Region', 'n_ints']].groupby('Region').median().sort_values(by='n_ints', ascending=False).index
    sns.boxplot(x='Region', y='n_ints', data=df, order=order, color='white', showfliers=False)
    sns.stripplot(x='Region', y='n_ints', data=df, order=order, jitter=0.2, palette=sns.color_palette('Paired'), alpha=0.5)
    ax.set_ylabel("Number of step intervals")

    fig.savefig(PATH_FIG.joinpath(f"si_scale_size_reg.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_scale_size_reg.png"), bbox_inches='tight')


################################################################
### SI Fig S23
### Number of scale degrees per scale, shown per scale type

def scale_size_per_st(df):
    fig, ax = plt.subplots(3,1,figsize=(9,6))
    fig.subplots_adjust(hspace=0.5)
    xlim = [np.array([-0.7, 23.7]) - x  for x in [0, 2, 4]]
    for i, st in enumerate(SCALE_TYPE):
        sns.countplot(x='n_ints', data=df.loc[df.ScaleType==st], ax=ax[i])
        ax[i].set_xlabel("Number of step intervals")
        ax[i].set_title(st, loc='left', fontsize=10)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlim(xlim[i])

    fig.savefig(PATH_FIG.joinpath(f"si_scale_size_st.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_scale_size_st.png"), bbox_inches='tight')


################################################################
### SI Fig S26
### Log-likelihood ratio for harmonoicity and interference models

def fig3c_interference(redo=False, ax='', mode='A', nsamp_max="o20"):
    if isinstance(ax, str):
        fig = plt.figure(figsize=(14,10))
        gs = GridSpec(3,2,width_ratios=[2,1])
        ax1 = fig.add_subplot(gs[:,0])
        ax = [ax1, ax1, ax1] + \
             [fig.add_subplot(gs[j,i]) for i in [1] for j in [0,1,2]]
        fig.subplots_adjust(hspace=0.5)

    df = scales_io.load_all_data()
    weights = WS.get_weights_by_region(df)
    weightsR = WS.get_weights_within_regions(df)
    weightsO = WS.get_weights_by_region_overall(df)
    regions = np.array(sorted(df.Region.unique()))

    # Pre-load data / save for future use
    path_data = PATH_FIG_DATA.joinpath(f"si_fig3c_data_{mode}_{nsamp_max}.pkl")
    if path_data.exists() and not redo:
        best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open(path_data, 'rb'))
    else:
        best_models = {}
        Yn, reg0, Yreg, wsum_list = [{st:[] for st in SCALE_TYPE} for _ in range(4)]
        for i, st in enumerate(SCALE_TYPE):
            # Find the best models, and save both LLR and model indices
            best_models[st] = [EM.best_model_across_n(m, st, nsamp_max, mode) for m in HI_MODELS]
            
            # Best-fitting model indices
            idx0 = {m: best_models[st][j][1][0] for j, m in enumerate(HI_MODELS)}

            # Get LLR per N (model indices are fixed)
            for j, m in enumerate(HI_MODELS):
                # Using the best-fitting model indices, get LLR for each N
                Yn[st].append(EM.best_model_across_n(m, st, nsamp_max, 'D'+mode, sep_n=1, idx0=idx0[m])[2])

            # Get LLR per Region (model indices are free)
            for j, reg in enumerate(regions):
                ist, insamp = EM.get_weight_index(st, reg)
                wsum = np.sum([weightsR[n][ist,insamp].sum() for n in EM.N_INTS])
                if wsum >= 10:
                    reg0[st].append(reg)
                    wsum_list[st].append(wsum)
                    Yreg[st].append([EM.best_model_across_n(m, st, reg, mode)[0] for m in HI_MODELS])
            Yreg[st] = np.array(Yreg[st])
        data = [best_models, Yn, reg0, Yreg, wsum_list]
        pickle.dump(data, open(path_data, 'wb'))

    X = np.arange(len(HI_MODELS))
    col = np.array(sns.color_palette('Paired'))[[1,3,5]]
    col_key = {r:c for c, r in zip(sns.color_palette('Paired'), regions)}
    xlim = (-1, 5)
    al = 0.7
    fs = 12
    pat = ['-', '--', ':']
    for i, st in enumerate(SCALE_TYPE):
        # Plot LLR
        width = 0.25
        xoff = (i - 1) * width
        Y = np.array([y[0] for y in best_models[st]])
        ax[i].bar(X + xoff, Y, width, color=COL_ST[st], alpha=al, ec='k', lw=0.5)
        print(Y)

        # Plot null model adjustment for fitting a free variable
        ist, insamp = EM.get_weight_index(st, nsamp_max)
        if isinstance(nsamp_max, int):
            wsum = np.sum([weights[n][ist,insamp].sum() for n in EM.N_INTS])
        else:
            wsum = weightsO[ist,insamp].sum()
        if mode == 'A':
            null_2sigma = np.array([2 / (wsum)**0.5] * X.size)
        elif mode == 'B':
            # Need to take into account fitting separate values of beta,
            # which depends on how many types of n_ints values are present in the set
            if isinstance(nsamp_max, int):
                n_beta_vals = np.sum([np.any(weights[n][ist,insamp] > 0) for n in EM.N_INTS])
            else:
                n_beta_vals = np.sum(np.any(weightsO[ist,insamp] > 0))
            null_2sigma = np.array([2 * (n_beta_vals / wsum)**0.5] * X.size)

        idx = null_2sigma < Y
        ax[i].plot(X[idx] + xoff, Y[idx] * 1.15, '*k')

        idx = (null_2sigma * 1.5) < Y
        ax[i].plot(X[idx] + xoff, Y[idx] * 1.4, '*k')

        ax[i].set_xticks(X)
        ax[i].set_xticklabels(HI_NAMES)

        # Plot LLR vs Region
        n_reg = len(reg0[st])
        width = 0.8 / n_reg
        for j, (y, r) in enumerate(zip(Yreg[st], reg0[st])):
            xoff = (j - n_reg/2 + 0.5) * width
            ax[3+i].bar(X + xoff, y, width, color=col_key[r], label=r, alpha=al, ec='k', lw=0.5)

            if mode == 'A':
                null_2sigma = 2 / np.array(wsum_list[st][j])**0.5
            elif mode == 'B':
                # Need to account for fitting separate values of beta
                ist, insamp = EM.get_weight_index(st, r)
                n_beta_vals = np.sum([np.any(weightsR[n][ist,insamp] > 0) for n in EM.N_INTS])
                null_2sigma = 2 * (n_beta_vals / wsum_list[st][j])**0.5

            idx1 = null_2sigma < y
            idx2 = (null_2sigma * 1.5) < y

            if i == 0:
                y1 = y[idx1] + 0.05
                y2 = y[idx2] + 0.12
            elif i == 1:
                y1 = y[idx1] + 0.10
                y2 = y[idx2] + 0.23
            elif i == 2:
                y1 = y[idx1] * 1.15
                y2 = y[idx2] * 1.45

            ax[3+i].plot(X[idx1] + xoff, y1, '*k', ms=4, fillstyle='full')
            ax[3+i].plot(X[idx2] + xoff, y2, '*k', ms=4, fillstyle='full')

        ax[3+i].set_xticks(np.arange(len(HI_NAMES)))
        ax[3+i].set_xticklabels(HI_NAMES, rotation=30)
        ax[3+i].legend(frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.60, 1.15), fontsize=fs-4,
                       handlelength=0.8, columnspacing=0.8, handletextpad=0.4)
        ax[3+i].set_ylabel(r"Log-likelihood Ratio")
        ax[3+i].set_title(st, loc='left', fontsize=fs)
    ax[5].set_yscale('log')

    handles = [Patch(ec='k', fc=COL_ST[st], alpha=al) for st in SCALE_TYPE]
    lbls = list(SCALE_TYPE)
    ax[0].legend(handles, lbls, ncol=4, frameon=False, fontsize=fs,
                   loc='upper center', bbox_to_anchor=(0.5, 1.05))
    ax[0].set_ylabel(r"LLR")
    ax[0].set_yscale('log')

    fs = 12
    ax[0].text(-.05, 1.01, 'A', transform=ax[0].transAxes, fontsize=fs)
    ax[3].text(-.10, 1.05, 'B', transform=ax[3].transAxes, fontsize=fs)
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath(f"si_fig3c_interference.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_fig3c_interference.png"), bbox_inches='tight')
    

################################################################
### SI Fig S27
### Scale degree predictions for harmonoicity and interference models

def fig4_interference():
    fig = plt.figure(figsize=(12,10))
    gs = GridSpec(5,3, height_ratios=[1.5, 0.01, 1, 1, 1])
    ax = [fig.add_subplot(gs[i,j]) for i in [0,2,3,4] for j in range(3)]
    fig.subplots_adjust(wspace=0.3, hspace=0.7)
    fig4a(ax[:3])
    fig4b(np.array(ax[3:]).reshape(3,3))
    fs = 12
    for i, (a, b) in enumerate(zip(ax[::3], 'ABCD')):
        a.text(-.10, 1.05, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si_fig4_interference.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_fig4_interference.png"), bbox_inches='tight')
    

def fig4a(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,3,figsize=(12,6))

    best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open('../Figures/Data/si_fig3c_data_A_o20.pkl', 'rb'))
    params = CF.get_model_params()
    off = 0
    for i, st in enumerate(SCALE_TYPE):
        for j in [1, 0, 2, 3, 4, 5]:
            m = HI_MODELS[j]
            name = "_".join(x for x in [m] + [str(p[k]) for p, k in zip(params[m], best_models[st][j][1][0][3:])])
            print(name)
            X, Y = paper_figs.get_model_score(name)
            ax[i].plot(X/100, Y + off, label=HI_NAMES[j])
            off += np.max(Y)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ylim = ax[i].get_ylim()
        for x in range(1, 12):
            ax[i].plot([x,x], ylim, ':', c='grey', alpha=0.5)
        ax[i].set_ylim(ylim)
        ax[i].set_xlabel("Interval (semitones)")
        ax[i].set_ylabel("HI Score")
        ax[i].set_title(st, loc='left')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.50, 1.35), ncol=6, frameon=0,
                 handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    

def fig4b(ax='', redo=False, n_ints=7):
    X = np.arange(0, 1751)
    path = PATH_FIG_DATA.joinpath("interference_genpop_kde.npy")
    if path.exists() and not redo:
        kde = np.load(path)
    else:
        ints = GS.load_all_res_int()
        jsd = GS.load_all_res_int('jsd')
        kde = []
        for i, st in enumerate(SCALE_TYPE):
            for j, m in enumerate(HI_MODELS):
                i0 = np.argmax(jsd[st][m][:,:,i].mean(axis=1))
                kde.append(gaussian_kde(np.cumsum(ints[st][m][i0], axis=2).ravel(), bw_method=0.05)(X))
        kde = np.array(kde).reshape(len(SCALE_TYPE), len(HI_MODELS), len(X))
        np.save(path, kde)


    if isinstance(ax, str):
        fig, ax = plt.subplots(3,3,figsize=(12,8))
        fig.subplots_adjust(wspace=0.2, hspace=0.5)
    xmax = [15, 15, 13]
    col = sns.color_palette()
#   kde = kde[:,3:]

    for i, st in enumerate(SCALE_TYPE):
        Xref, Yref = np.load(PATH_COMP.joinpath(f"ScaleDegreeDist/{st}_{n_ints}_20.npy")).T
        for j in range(3):
            ax[j,i].plot(Xref/100, Yref, '-k', label='Empirical')
            for k in [1, 0, 2]:
                ax[j,i].plot(X/100, kde[i,k], label=HI_NAMES[k], alpha=0.7)

        for j, m in enumerate(HI_MODELS[3:]):
            ax[j,i].plot(X/100, kde[i,j+3], label='Interference', c=col[j+3])

            ylim = ax[j,i].get_ylim()
            for x in range(1, xmax[i]):
                ax[j,i].plot([x,x], ylim, ':', c='grey', alpha=0.5)
            ax[j,i].set_ylim(ylim)
            ax[j,i].set_xlim(0,16)
            ax[j,i].spines['top'].set_visible(False)
            ax[j,i].spines['right'].set_visible(False)
            ax[j,i].set_yticks([])
            ax[j,i].set_ylabel("Density")
            ax[j,i].set_xlabel("Scale Degree (semitones)")
            ax[j,i].set_title(st, loc='left')


################################################################
### SI Fig S24
### Melodic interval distributions for 62 melodic corpora

def melody_corpus_scale_degree():
    fig, ax = plt.subplots(9,7,figsize=(12,15))
    ax = ax.reshape(ax.size)
    fig.subplots_adjust(hspace=1.1, wspace=0.6)
    count = np.loadtxt(PATH_DATA.joinpath("Melodies", "folk_corpora_scale_degree_counts.txt"))
    countries = np.loadtxt(PATH_DATA.joinpath("Melodies", "folk_corpora_groups.txt"))
    X = np.arange(1, 13)
    col = sns.color_palette('husl', n_colors=12)
    for i, (c1, c2) in enumerate(zip(count, countries)):
        ax[i].bar(X, c1, color=col)
        ax[i].set_xlabel("\# Scale Degrees")
        ax[i].set_ylabel("Count")
        ax[i].set_title(c2)

    fig.savefig(PATH_FIG.joinpath(f"si_melody_scale_degree.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath(f"si_melody_scale_degree.png"), bbox_inches='tight')



#####################################
### SI Fig S25
### Complexity model figures

### Requires access to a file that is not public.
### All of the required data is available at:
###     https://zenodo.org/records/10608046
### One needs to collect all of the scale data in SupplementaryData.zip
### and extract the means (df.ScaleCents) and weights (df.weights),
### and put it into a pandas dataframe format. 
def tonality():
    path_to_HSSC_dataset = Path()
    df = pd.read_pickle(path_to_HSSC_dataset)
    cd, w2 = np.array([[x, y] for c, w in zip(df.ScaleCents, df.weights) for x, y in zip(*get_combined_weight(c, w))]).T

    cdi = (cd - 25)//50
    X2 = np.arange(25, 1850, 50) 
    Y2 = np.array([np.nanmean(w2[cdi==i]) for i in range(X2.size)])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot((X2 + 25) / 100, Y2) 
    ax.set_xlabel("Interval (semitones)")
    ax.set_ylabel("Tonal hierarchy weight")

    ylim = [0, 0.035]
    for x in range(1, 18): 
        ax.plot([x,x], ylim, ':', c='grey', alpha=0.5)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("si_tonal_hierarchy.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si_tonal_hierarchy.png"), bbox_inches='tight')


def get_combined_weight(C, W): 
    dc = np.diff(np.meshgrid(C, C), axis=0)[0]
    w2 = np.outer(W, W)
    idx = np.triu_indices(len(W), 1)
    return np.abs(dc[idx]), w2[idx]


################################################################
### SI Table S1
### Melodic interval distributions for 62 melodic corpora

def best_params(redo=False, mode='A', nsamp_max="o20"):
    df = scales_io.load_all_data()
    df = df.loc[df.n_ints.isin(EM.N_INTS)]
    
    path = PATH_FIG_DATA.joinpath(f"best_params.pkl")
    if path.exists() and not redo:
        param_idx = pickle.load(open(path, 'rb'))
    else:
        param_idx = {}
        for j, st in enumerate(SCALE_TYPE):
            ist, insamp = EM.get_weight_index(st, nsamp_max)
            otxt = 'octave' if st == 'Theory' else '1250'
            weightsO = WS.get_weights_by_region_overall(df)[ist, insamp]
            param_idx_model = {}

            for model in ['H_GFO', 'H_GP', 'H_HP', 'C_NI']:
                sprob_n = {n: np.load(f'../Precompute/CostFn/ScaleProb/{model}_{otxt}_{n}.npy') for n in EM.N_INTS}
                sprob0 = np.zeros(sprob_n[5].shape[:-1] + (len(df), ), float)
                for n in EM.N_INTS:
                    if len(sprob0.shape) == 6:
                        sprob0[:,:,:,:,:,df.n_ints==n] = sprob_n[n]
                    elif len(sprob0.shape) == 5:
                        sprob0[:,:,:,:,df.n_ints==n] = sprob_n[n]
                    elif len(sprob0.shape) == 4:
                        sprob0[:,:,:,df.n_ints==n] = sprob_n[n]
                sprob0[~np.isfinite(sprob0)] = np.nan

                mprob0 = np.average(np.log(sprob0), weights=weightsO, axis=-1)

                idx = np.unravel_index(np.nanargmax(mprob0), mprob0.shape)
                param_idx_model[model] = idx
            param_idx[st] = param_idx_model
        pickle.dump(param_idx, open(path, 'wb'))

    names = EM.updated_param_names()
    params = EM.get_updated_model_param_vals()
    beta_list = GS.selected_models()[2]
    data = []
    model1 = ['Harmony'] * 3 + ['Complexity'] 
    model2 = ['OF', 'GP', 'HP', '']
    cols = ['Theory', 'Version', 'Scale Type', 'Best-fitting Parameters']
    name_key = {'W_ARR': r"$$w$$", 'W': r"$$w$$", 'N':r"$$n$$", 'rho':r"$$\rho$$", 'n':r"$$n$$"}

    for m, m1, m2 in zip(['H_GFO', 'H_GP', 'H_HP', 'C_NI'], model1, model2):
        for st in SCALE_TYPE:
            off = 1 if m == 'C_NI' else 2
            n0 = [r'$$\Beta$$'] + [name_key[n] for n in names[m][off:]]
            p0 = [beta_list] + params[m][off:]

            exc = [1] if m == 'C_NI' else [1,2]
            i0 = [j for i, j in enumerate(param_idx[st][m]) if i not in exc]
            pstr = ', '.join([f"{n0[i]} = {p0[i][j]}" for i, j in enumerate(i0)])
            print(f"{m1}, {m2}, {pstr}")
            data.append([m1, m2, st, pstr])

    df = pd.DataFrame(columns=cols, data=data)
    open(PATH_FIG.joinpath("tab1.txt"), 'w').write(df.to_latex(index=False, float_format="{:.1f}".format))

    




