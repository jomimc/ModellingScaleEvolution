"""
This module, "paper_figs.py", contains code to reproduce figures from the main paper.

The figures are in principle reproducible,
however some of them require precomputation of cached data,
which is performed when running other code provided here, e.g., evaluate_model.py.
"""
from pathlib import Path
import pickle
import time

from brokenaxes import brokenaxes
import geopandas
from shapely.geometry.point import Point
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.optimize import brute, minimize
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist
from scipy.stats import norm, pearsonr, gaussian_kde, ks_2samp, entropy
import seaborn as sns

import cost_functions as CF
import interval_constraints as IC
import interval_stats as IS
import evaluate_models as EM
import generate_scales as GS
from scales_config import *
import scales_io
import scales_utils as utils
import weighted_sampling as WS


plt.rcParams['text.usetex'] = True
COL_REG = {r:c for r, c in zip(REGIONS, sns.color_palette('Paired'))}
COL_ST = {r:sns.color_palette('colorblind')[i] for r, i in zip(SCALE_TYPE, [2,3,4])}


################################################################################
### Fig 1
### This section contains parts of the instructional figure, that are combined elswhere


def fig1_interference(f0=440):
    fig, ax = plt.subplots(figsize=(5,1.5))
    X = np.linspace(0, 0.5, 10001)
    Y1 = np.sin(X * f0 * np.pi)
    Y2 = np.sin(X * f0 * np.pi * 1.059)
    Y3 = Y1 + Y2
    offset = [8, 5, 1]
    col = sns.color_palette()[:2] + [sns.color_palette()[2]]
    for i, (y, o) in enumerate(zip([Y1, Y2, Y3], offset)):
        ax.plot(X * 1000, y + o, c=col[i])


    # Get interference periodicity
    imax = argrelmax(Y3)[0]
    y0 = 7.8
    period = np.mean([x * 1000 for x in np.diff(X[imax][Y3[imax]>1.99]) if x > 0.01])

    ax.set_xlim(0, 200)
    ax.set_ylim(-1, 10.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks(offset)
    ax.set_yticklabels([r"$f_1$ = 440 Hz", r"$f_2$ = 466 Hz", r"$f_1 + f_2$"])
    ax.set_xticks([])
    ax.tick_params(length=0)

    fig.savefig(PATH_FIG.joinpath("fig1_interference.pdf"), bbox_inches='tight')


def fig1_fit_flat():
    fig, ax = plt.subplots(figsize=(3,1.5))
    ax2 = ax.twinx()
    ax.set_xlabel("Trait")
    ax.set_ylabel("Fitness")
    ax2.set_ylabel("")


def fig1_IS():
    fig, ax = plt.subplots(figsize=(3,1.5))
    X = np.linspace(-3, 6, 1000)
    Y1 = norm.pdf(X, 0, 1)
    Y2 = norm.pdf(X, 3, 1)
    ax.plot(X, Y1, '-k')
    ax.plot(X, Y2, ':k')
    ax.set_xticks([0,3])
    ax.set_xticklabels(['A', 'B'])
    ax.set_xlabel("Interval")
    ax.annotate("", xy=(0, 0.45), xytext=(3, 0.45), arrowprops=dict(arrowstyle='<->', lw=1.5, ls='-'))
    ax.text(1.3, 0.48, r"$\Delta I$")
    ax.set_ylim(0, .5)
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("fig1_IS.pdf"), bbox_inches='tight')


def fig1_MC():
    fig, ax = plt.subplots(figsize=(3,1.5))
    fig.subplots_adjust(wspace=0.4)
    X = np.linspace(0, 1, 1000)
    Y1 = X**2
    Y2 = np.exp(-X*3)
    ax.plot(X, Y1, '-k')

    ax.set_ylabel("Energy")
    ax.set_xlabel(r"$\Delta I$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(PATH_FIG.joinpath("fig1_MC.pdf"), bbox_inches='tight')

    
def fig1_har():
    fig, ax = plt.subplots(figsize=(5,1.5))
    X = np.linspace(-3, 10, 1000)
    col = sns.color_palette()
    for x in range(1, 8):
        for j, y in enumerate([x, x*2]):
            Y = norm.pdf(X, y, 0.1) * 10**((8 - x)/5)
            idx = Y > 0.1
            ax.plot(X[idx], Y[idx], c=col[j])
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel("Harmonics")

    handles = [Line2D([], [], ls='-', c=col[i]) for i in range(3)]
    ax.legend(handles, ['A', 'B'], frameon=0, ncol=1, bbox_to_anchor=(0.80, 1.00), loc='upper left')

    fig.savefig(PATH_FIG.joinpath("fig1_har.pdf"), bbox_inches='tight')


################################################################################
### Fig 2


def fig2_world(df, jitter=0):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world['longitude'] = world.centroid.apply(lambda x: x.x)
    world['latitude'] = world.centroid.apply(lambda x: x.y)

    # Rename countries so that they fit with geopandas
    df = df.copy()
    df = utils.rename_countries(df)
    df = utils.add_latlong(df)
    print(len(df['country_map'].unique()))

    # Assign lat / long values
    key = {k:v for k, v in zip(world.name, world.latitude)}
    df.loc[df.latitude.isnull(), 'latitude'] = df.loc[df.latitude.isnull(), 'country_map'].map(key)

    key = {k:v for k, v in zip(world.name, world.longitude)}
    df.loc[df.longitude.isnull(), 'longitude'] = df.loc[df.longitude.isnull(), 'country_map'].map(key)

    df.loc[(df.Country=='Many')&(df.Region=='ME'), 'longitude'] = 45
    df.loc[(df.Country=='Many')&(df.Region=='ME'), 'latitude'] = 24

    df.loc[(df.Country=='Many')&(df.Region=='Eu'), 'longitude'] = 10
    df.loc[(df.Country=='Many')&(df.Region=='Eu'), 'latitude'] = 47

    df.loc[(df.Country=='Many')&(df.Region=='Af'), 'longitude'] = -4
    df.loc[(df.Country=='Many')&(df.Region=='Af'), 'latitude'] = 17

    # Add offsets to distinguish scale types
    off = 1
    df.loc[df.ScaleType=='Instrumental', 'latitude'] = df.loc[df.ScaleType=='Instrumental', 'latitude'] - off
    df.loc[df.ScaleType=='Instrumental', 'longitude'] = df.loc[df.ScaleType=='Instrumental', 'longitude'] - off
    df.loc[df.ScaleType=='Vocal', 'latitude'] = df.loc[df.ScaleType=='Vocal', 'latitude'] + off
    df.loc[df.ScaleType=='Vocal', 'longitude'] = df.loc[df.ScaleType=='Vocal', 'longitude'] + off
    df.loc[df.ScaleType=='Theory', 'latitude'] = df.loc[df.ScaleType=='Theory', 'latitude'] + off
    df.loc[df.ScaleType=='Theory', 'longitude'] = df.loc[df.ScaleType=='Theory', 'longitude'] - off * 1.5

    # Convert lat/long to Points, and add jitter
    if jitter > 0:
        df.longitude = df.longitude + np.random.normal(0, jitter, size=len(df))
        df.latitude = df.latitude + np.random.normal(0, jitter, size=len(df))
    df['cent_col'] = [Point(x, y) for x, y in zip(df.longitude, df.latitude)]
    

    # Plot world map
    fig, ax = plt.subplots()
    ec = 0.7
    world.plot(ax=ax, color=(1.0, 1.0, 1.0), edgecolor=(ec, ec, ec), lw=0.5)
    col = sns.color_palette()
    marker = ['o', 's', '^']
    marker = 'ooo'

    for i, st in enumerate(SCALE_TYPE):
        gdf = geopandas.GeoDataFrame(pd.DataFrame(data={'coord': df.loc[df.ScaleType==st, 'cent_col'].values}), geometry='coord')
        gdf['ms'] = 5
        gdf.plot(ax=ax, markersize=gdf['ms'], alpha=1, color=COL_ST[st], facecolors='none', lw=0.7, marker=marker[i])

    handles = [Line2D([], [], ls='', marker=marker[i], c=COL_ST[st], fillstyle='none', ms=7) for i, st in enumerate(SCALE_TYPE)]
    ax.legend(handles, SCALE_TYPE, frameon=0, ncol=3, bbox_to_anchor=(0.15, 1.15), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(PATH_FIG.joinpath("fig1_map.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig1_map.png"), bbox_inches='tight')



################################################################################
### Fig 3


def fig3(df):
    fig = plt.figure(figsize=(12,10))
    gs = GridSpec(6,6, height_ratios=[1, 1, 1.3, 1.3, 1.3, 1.3])
    ax = [fig.add_subplot(gs[:2,:2]), fig.add_subplot(gs[:2,2:])] + \
         [fig.add_subplot(gs[2:4,:3]), fig.add_subplot(gs[2:4,3:])] + \
         [fig.add_subplot(gs[4:6,:4]), fig.add_subplot(gs[4,4]), fig.add_subplot(gs[4,5]), fig.add_subplot(gs[5,4:])]
    fig.subplots_adjust(wspace=0.8, hspace=1.00)
    fig3a(df, ax[0])
    fig3b(df, ax[1])
    fig3cd(df, ax[2:4])
    fig3e(df, ax[4])
    fig3fg(ax[5:7])
    fig3h(ax[7])

    fs = 12
    xi = [(-.15, 1.08), (-.06, 1.08), (-0.08, 0.98), (-0.08, 0.98), (-.10, 1.03), (-.20, 1.03), (-.20, 1.03), (-.15, 1.03)]
    for i, (a, b) in enumerate(zip(ax, 'ABCDEFGH')):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.text(*xi[i], b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("fig3.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig3.png"), bbox_inches='tight')


def fig3a(df, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    dfi = IC.get_interval_df(df)
    sns.boxenplot(x='ScaleType', y='Int', data=dfi, ax=ax, showfliers=False, palette=COL_ST)
    ax.set_ylabel(r"Step Interval, $I_S$ (semitones)")
    ax.set_xlabel('')


def fig3b(df, ax='', st=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    if len(st):
        df = df.loc[df.ScaleType==st]
    df = utils.rename_regions(df)
    dfi = IC.get_interval_df(df)
    reg_full = [s.replace(' ', '\n') for s in REG_FULL]
    reg_key = {r1: r2 for r1, r2 in zip(REGIONS, reg_full)}
    dfi['Region'] = dfi['Region'].map(reg_key)

    int_median = dfi.groupby('Region')['Int'].median().sort_values()
    order = dfi.groupby('Region')['Int'].mean().sort_values().index
    sns.boxenplot(x='Region', y='Int', data=dfi, ax=ax, showfliers=False, order=order, palette='husl')
    ax.set_ylim(0, 5)
    ax.set_ylabel(r"Step Interval, $I_S$ (semitones)")
    ax.set_xlabel('')


def fig3cd(df, ax='', st='Vocal', n_rep=10000, scale_range=1700):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(8,4))

    df = df.loc[df.ScaleType==st]
    N = np.arange(2, 11)
    step = np.array([x for y in df.Ints for x in y])

    # Get Harmony predictions
    minmax = GS.step_int_minmax()

    fn_list = [np.min, np.max]
    pat = ['-o', '-s', ':^']
    lbls = ['Null', 'Empirical', 'Melody']
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
#       ax[i].plot(N[:8], minmax[4,54,:,i] / 100, ':>', fillstyle='none', ms=10, label='Complexity', c=col[4])

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


def fig3e(df, ax='', st='Vocal', c0=207.4, octave=False):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    step = np.array([x for y in df.loc[(df.n_notes>1)&(df.ScaleType==st), "Ints"] for x in y])
    X = np.arange(0, 1201)
    Y0 = gaussian_kde(step)(X)
    ax.plot(X / 100, Y0, '-k')

    X = np.arange(1201) / 100
    beta_list = GS.selected_models()[2]
    i0_list = [2]#, 3]
    j0_list = [2]#, 4]
    k0 = [50, 54]
    n = 7
    lbls = ['Harmony']#, 'Complexity']

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


def fig3fg(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(8,4))

    folk_idx = pd.read_csv('../Data/Melodies/melodic_database_summary.csv')['ctype'] == 'Folk'
    int_prob = np.load(PATH_DATA.joinpath("Melodies", 'melodic_interval_dist_per_corpus.npy'))[folk_idx]
    coef = np.load(PATH_DATA.joinpath("Melodies", 'melodic_interval_scaling_coef_per_corpus.npy'))[folk_idx]

    # First entry corresponds to the Jamaica corpus
    X = np.arange(0, 15)
    ax[0].plot(X, int_prob[0] / np.nansum(int_prob[0]), 'o', color='grey')

    Y = np.exp(-X / coef[0])
    ax[0].plot(X, Y / Y.sum(), '-k', label=r"$P_{\textrm{MC}} \propto e^{-I/I_{0}}$")

    sns.histplot(x=coef, color='grey', ax=ax[1])

    ax[0].set_yscale('log')
#   ax[0].set_xlabel("Melodic Interval (semitones)")
    ax[0].set_xlabel(r"$I$ (semitones)")
    ax[0].set_ylabel("Probability")
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), frameon=False,
                 handlelength=0.5, handletextpad=0.4, fontsize=8)
    ax[1].set_xlabel(r"$I_{0}$ (semitones)")


def fig3h(ax='', s=31, L=14):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    X = np.arange(0, 255, 5)
    ax.plot(X / 100, norm.cdf(X/2, 0, s)**L)
    ax.set_xlabel(r"$I$ (semitones)")
    ax.set_ylabel(r"$P_{\textrm{IS}}(I)$")




################################################################################
### Fig 4

def fig4(df, redo=False, nsamp_max="o20", modeA='resample', modeC='A'):
    fig = plt.figure(figsize=(12,12))

    gs = GridSpec(22,13)
    ax = [fig.add_subplot(gs[i:j,:6]) for i, j in [(0,4), (5,9), (10,14)]] + \
         [fig.add_subplot(gs[i:j,7:]) for i, j in [(0,4), (5,9), (10,14)]] + \
         [fig.add_subplot(gs[16:,i:j]) for i, j in [(0,3),(3,6)]] + \
         [fig.add_subplot(gs[16:,7:])]

    fig.subplots_adjust(wspace=2.0, hspace=4.0)

    fig4a(df, ai=1, mode=modeA, ax=ax[:3])
    fig4c(ax[3:6], 7)
    fig4b(redo, ax[6:8], modeC, nsamp_max)
    fig4d(ax=ax[8:])

    fs = 12
    for i, a in enumerate(ax):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for i, b in zip([0, 3, 6, 8], 'ACBD'):
        ax[i].text(-.10, 1.07, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("fig4.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig4.png"), bbox_inches='tight')


def fig4a(df, single=False, bw=20, xmax=1780, ai=False, i2=4, mode='resample', ax=''):
    regions = np.array(sorted(df.Region.unique()))

    i1 = 1 if ai else 0

    path_data = scales_io.PATH_RES.joinpath("IntervalStats")
    count = [np.mean(np.load(path_data.joinpath(f"int_prob_lognorm_{st}.npy"))[i1,i2,:,0], axis=0) for st in SCALE_TYPE]

    prob_data = IS.get_significant_intervals_st(i1, i2, mode)
    prob = prob_data[:,0]
    is_less = prob_data[:,1].astype(bool)

    bins = np.arange(10, xmax+bw, bw)
    X = bins[:-1] + bw//2
    offset = [0.0050, 0.0030, 0.0100]
    X = X / 100

    if isinstance(ax, str):
        fig, ax = plt.subplots(3,1,figsize=(8,5))

    fs = 12
    star_lbls = [[f"{a}; {b}" for a in ['Common', 'Uncommon']] for b in [r'$p < 0.05$', r'$p < 0.005$']]
    ylim = [0.055, 0.048, 0.16]
    xmax = [18, 18, 13]
    for i, st in enumerate(SCALE_TYPE):
        for x in range(1, xmax[i]):
            ax[i].plot([x,x], [0, ylim[i]], ':k', alpha=0.3)

        hist = count[i] / np.sum(count[i])
        if st != 'Theory':
            mask = np.ones(X.size, bool)
        else:
            mask = X < 12
        ax[i].bar(X[mask], hist[mask], bw / 100, ec='k', alpha=0.7, color=COL_ST[st], lw=0.3)
        for j, is_sig in enumerate([prob[i] < 0.05, prob[i] < 0.005]):
            for idx, fill, l in zip([~is_less[i], is_less[i]], ['full', 'none'], star_lbls[j]):
                ax[i].plot(X[is_sig & idx & mask], hist[is_sig & idx & mask] + offset[i] * (j+1), '*k', fillstyle=fill, mew=0.3, ms=5)#, label=l)

        # Plot Melody model distribution
        mel_prob, mel_ci = get_melody_int_prob(st, i1, i2)
        ax[i].plot(X, mel_prob, '-', color='grey', label='Melody model')
        ax[i].fill_between(X, *mel_ci, color='grey', alpha=0.4, label=r'Melody model 95\% CI')
        

        ax[i].set_xlim(0, 17.3)
        ax[i].set_title(st, loc='left', fontsize=fs)
        ax[i].set_xlabel(r"Scale Interval, $I_A$ (semitones)")
        ax[i].set_ylabel("Density")

        ax[i].set_ylim(0, ylim[i])
        ax[i].set_xticks(range(0, xmax[i], 2))
        ax[i].set_yticks([])

    xi = [13, 13.2, 13.5]
    xi_list = [xi[:1], xi[:2], xi[:1], xi[:2]]
    fill = ['full'] * 2 + ['none'] * 2
    txt = [f'p $<$ {a} ({b})' for b in ["common", "rare"] for a in  [0.05, 0.005]]
    for i in range(4):
        yi = 0.14 - 0.02 * i
        ax[2].plot(xi_list[i], [yi] * len(xi_list[i]), '*k', fillstyle=fill[i], mew=0.3, ms=5)
        ax[2].text(xi[2], yi - 0.005, txt[i])
    ax[2].legend(bbox_to_anchor=(0.75, 0.10), frameon=False, handletextpad=0.5)
    
    
def get_melody_int_prob(st, i1, i2, reg='', mode='resample'):
    path_data = scales_io.PATH_RES.joinpath("IntervalStats")
    data = np.load(path_data.joinpath(f"int_prob_lognorm_{st}{reg}_{mode}.npy"))[i1,i2,:]
    count = data[:,0]
    N = data[:,1]
    prob = np.mean(count / N, axis=0)
    ci = np.quantile(count / N, [0.025, 0.975], axis=0)
    return prob, ci


def fig4c(ax='', n_ints=7):
    if isinstance(ax, str):
        fig, ax = plt.subplots(3,1,figsize=(12,5))

    jsd = np.array([GS.load_results(i, 'jsd', 0, n_ints) for i in range(6)]).mean(axis=2)
    jsd_octave = np.array([GS.load_results(i, 'jsd', 1, n_ints) for i in range(6)]).mean(axis=2)
    octave = [0,0,1]
    xmax = [15, 15, 13]
    idx = [2,2,3]
    for i, st in enumerate(SCALE_TYPE):
        j = idx[i]
        if st == 'Theory':
            imin = np.argmin(jsd_octave[j,:,i])
            Xref, Yref, Yhar = load_gen_pop_dist(i0=j, j0=imin, st=st, octave=1, n_ints=n_ints)
            Ymel = load_gen_pop_dist(i0=j, j0=0, st=st, octave=1, n_ints=n_ints)[2]
        else:
            imin = np.argmin(jsd[j,:,i])
            Xref, Yref, Yhar = load_gen_pop_dist(i0=j, j0=imin, st=st, octave=0, n_ints=n_ints)
            Ymel = load_gen_pop_dist(i0=j, j0=0, st=st, octave=0, n_ints=n_ints)[2]
        print(imin)

        ax[i].plot(Xref/100, Yref, '-k', label="Empirical")
        ax[i].plot(Xref/100, Ymel, label="Melody")
        ax[i].plot(Xref/100, Yhar, label="Harmony + Melody")


        ax[i].set_xlabel("Scale Degree (semitones)")
        ax[i].set_ylabel("Density")
        ax[i].set_yticks([])
        ax[i].set_title(st, loc='left', fontsize=10)
        ax[i].set_xlim(0,16)
        for x in range(1, xmax[i]):
            ylim = ax[i].get_ylim()
            ax[i].plot([x,x], ylim, ':', c='grey', alpha=0.5)
            ax[i].set_ylim(ylim)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.6, 1.28), ncol=3, frameon=0,
                       handlelength=1.5, columnspacing=1.0, handletextpad=0.4)


def fig4b(redo=False, ax='', mode='A', nsamp_max="o20"):
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

#   dfr = dfr.loc[dfr.model.isin(['H_HP', 'C_NI'])]
    dfr = dfr.loc[dfr.model == 'H_HP']

    sns.boxenplot(x='model', y='logprob', hue='ScaleType', data=dfr, showfliers=False, ax=ax[1], palette=COL_ST)

    xlim = ax[1].get_xlim()
    ax[1].plot(xlim, [0,0], ':k')
    ax[1].set_xlim(xlim)

#   X = np.arange(2)
    X = np.array([0])
    al = 0.7
    fs = 12
    width = 0.25
    pat = ['-', '--', ':']
    for i, st in enumerate(SCALE_TYPE):
        # Plot LLR
        xoff = (i - 1) * width
        Y = []
        for model in ['H_HP']:#, 'C_NI']:
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


    xlbls = ['Harmony +\nMelody']#, 'Complexity +\nMelody']
    for i, a in enumerate(ax):
        a.set_xticks(X)
        a.set_xticklabels(xlbls)
        a.set_xlabel('')
    ax[0].set_ylabel("Log-likelihood ratio per scale")
    ax[1].set_ylabel(r"Log-likelihood ratio")
    ax[1].legend(frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.15))



def fig4d(redo=False, ax='', mode='A', nsamp_max="o20"):
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
    ttls = ['Harmony + Melody']#, 'Complexity + Melody']
    for i, j in enumerate([2]):#,3]):
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

    handles = [Patch(facecolor=c, edgecolor='k', hatch=h, alpha=al) for c, h in zip(colors, hatch)]
    lbls = ['Too few data', r'$p > 0.05$', r'$p < 0.05$', r'$p < 0.005$', r'$p < 10^{-8}$']
    ax[0].legend(handles, lbls, loc='upper left', bbox_to_anchor=(0.40, 1.25),
                 frameon=False, handlelength=0.9, ncol=3, columnspacing=1.0, fontsize=fs)

    return sig_arr
    

def map_sig_col(sig_arr, colors):
    col_arr = np.zeros(sig_arr.shape + (3,), float)

    col_arr[np.isnan(sig_arr)] = colors[0]
    col_arr[sig_arr < 1] = colors[1]
    col_arr[(sig_arr >= 1) & (sig_arr < 1.5)] = colors[2]
    col_arr[(sig_arr >= 1.5) & (sig_arr < 3)] = colors[3]
    col_arr[sig_arr >= 3.0] = colors[4]

    return col_arr



################################################################################
### Fig 5

def fig5(df):
    fig = plt.figure(figsize=(10,7))

    gs = GridSpec(6,3, width_ratios=[1,.3,1.5])
    ax = [fig.add_subplot(gs[:3,0])] + \
         [fig.add_subplot(gs[i:i+2,2]) for i in [0,2,4]]

    bax = brokenaxes(ylims=((0, 1.5), (5, 9)), hspace=.25, subplot_spec=gs[3:,0])
    fig.subplots_adjust(wspace=0, hspace=1.4)

    fig5a(ax[0])
    fig5c(ax[1:4])
    fig5b(bax)

    fs = 12
    for i, a in enumerate(ax):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    for a in bax.axs:
        a.get_children()[0].set_alpha(0)
    bax.axs[0].tick_params(axis='y', which='minor')
    bax.axs[0].yaxis.set_minor_locator(MultipleLocator(0.5))
    bax.axs[1].yaxis.set_minor_locator(MultipleLocator(0.5))
    bax.axs[1].set_xticks(range(3))

    xi = [(-.06, 1.08), (-.06, 1.14), (-.35, 1.50), (-.15, 1.08)]
    for i, x, b in zip([0, 1, 4, 10], xi, 'AC'):
        ax[i].text(*x, b, transform=ax[i].transAxes, fontsize=fs)
    bax.axs[0].text(*xi[3], 'B', transform=bax.axs[0].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("fig5.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig5.png"), bbox_inches='tight')


def fig5a(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    models = ['H_GFO_20', 'H_GP_1_20', "H_HP_3_1", "H_HP_39_1"]
    lbls = ['OF', 'GP', r'HP$^{\textrm{A}}$', r'HP$^{\textrm{B}}$']
    for i, m in enumerate(models):
        X, Y = get_model_score(m)
        off = 3 - i
        ax.plot(X/100, Y + off, label=lbls[i])

    ylim = ax.get_ylim()
    for x in range(1, 12):
        ax.plot([x,x], ylim, ':', c='grey', alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xlabel("Interval (semitones)")
    ax.set_ylabel("Harmonicity Score")
    ax.set_xticks(range(0,14,2))
    ax.set_yticks([])
    ax.legend(loc='upper left', bbox_to_anchor=(0.10, 1.16), ncol=4, frameon=0,
              handlelength=1.5, handletextpad=0.5, columnspacing=1.0)
    

def get_model_score(model):
    X = np.arange(1250)
    i1 = (X >= 50)# & (X <= 1150)
    i2 = np.where(EM.NAMES == model)[0]
    Y = EM.SCORE[i2[0]]
    Y = Y / Y[702]
    return X[i1], Y[i1]


def fig5b(ax='', n_ints=7):
    if isinstance(ax, str):
        fig, ax = plt.subplots(3,1,figsize=(12,5))

    jsd = np.array([GS.load_results(i, 'jsd', 0, n_ints) for i in range(4)]).mean(axis=2)
    jsd_octave = np.array([GS.load_results(i, 'jsd', 1, n_ints) for i in range(4)]).mean(axis=2)
    octave = [0,0,1]
    xmax = [15, 15, 13]
    lbls = ['OF', 'GP', r'HP$^{\textrm{A}}$', r'HP$^{\textrm{B}}$']
    for i, st in enumerate(SCALE_TYPE):
        for k, j in enumerate([1,0,2,3]):
            if st == 'Theory':
                imin = np.argmin(jsd_octave[j,:,i])
                Xref, Yref, Ypop = load_gen_pop_dist(i0=j, j0=imin, st=st, octave=1, n_ints=n_ints)
                idx = Xref <= 1250
                Xref, Yref, Ypop = [x[idx] for x in [Xref, Yref, Ypop]]
            else:
                imin = np.argmin(jsd[j,:,i])
                Xref, Yref, Ypop = load_gen_pop_dist(i0=j, j0=imin, st=st, octave=0, n_ints=n_ints)
            print(st, j, round(pearsonr(Ypop, Yref)[0], 2))
            if k == 0:
                ax[i].plot(Xref/100, Yref, '-k', label="Empirical")
            ax[i].plot(Xref/100, Ypop, label=lbls[k])
        ax[i].set_xlabel("Scale Degree (semitones)")
        ax[i].set_ylabel("Density")
        ax[i].set_yticks([])
        ax[i].set_title(st, loc='left', fontsize=10)
        ax[i].set_xlim(0,16)
        for x in range(1, xmax[i]):
            ylim = ax[i].get_ylim()
            ax[i].plot([x,x], ylim, ':', c='grey', alpha=0.5)
            ax[i].set_ylim(ylim)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5, frameon=0,
                       handlelength=1.5, columnspacing=1.0, handletextpad=0.4)


def load_gen_pop_dist(i0=0, j0=40, st='Vocal', octave=False, nrep=10, n_ints=7, redo=False):
    otxt = "_octave" if octave else ""
    path = PATH_FIG_DATA.joinpath(f"fig4b_{i0}_{j0}_{st}_{n_ints}{otxt}.npy")
    if path.exists() and not redo:
        return np.load(path)

    cfn, params, beta_list = GS.selected_models()
    cfn, params = cfn[i0], params[i0]
    beta = beta_list[j0]

    name = f"{cfn}_" + "_".join(str(v) for v in params.values()) + f"_{np.log10(beta):05.2f}"
    scales = []
    for i in range(nrep):
        path_pop = PATH_COMP.joinpath("GenPop", f"ints_{name}_{n_ints}_{i:03d}{otxt}.npy")
        ints = np.load(path_pop)
        scales.extend(np.cumsum(ints, axis=1).ravel())
    kde_fn = gaussian_kde(scales, bw_method=0.05)
    Xref, Yref = np.load(PATH_COMP.joinpath(f"ScaleDegreeDist/{st}_{n_ints}_20.npy")).T
    Ypop = kde_fn(Xref)

    np.save(path, np.array([Xref, Yref, Ypop]))

    return Xref, Yref, Ypop


def multicolored_line(X, Y, ax, cm='flare'):
    N = len(X)
    cmap = sns.color_palette(cm, as_cmap=True)

    # Find where each point lies on the line as a function of cumulative
    # distance travelled, so that the colors are equally spaced
    line_distance = np.linalg.norm([X[:-1] - X[1:], Y[:-1] - Y[1:]], axis=0)
    progression = np.cumsum(np.abs(line_distance))
    progression = progression / progression[-1]
    col = cmap(progression)
    for i in range(0, N - 1):
        ax.plot(X[i:i+2], Y[i:i+2], c=col[i], lw=2)


def fig5b(ax='', mode='A', nsamp_max="o20"):
    if isinstance(ax, str):
        fig = plt.figure(figsize=(6, 4))
        ax = brokenaxes(ylims=((0, 1.6), (5, 9)), hspace=.25)

    path_data = PATH_FIG_DATA.joinpath(f"fig3c_data_{mode}_{nsamp_max}.pkl")
    best_models, Yn, reg0, Yreg, wsum_list = pickle.load(open(path_data, 'rb'))

    df = scales_io.load_all_data()
    weightsO = WS.get_weights_by_region_overall(df)

    X = np.arange(3)
    col = sns.color_palette()
    xlim = (-1, 4)
    al = 0.7
    fs = 12
    pat = ['-', '--', ':']
    for i, st in enumerate(SCALE_TYPE):
        # Plot LLR
        width = 0.25
        xoff = (i - 1) * width
        Y = np.array([y[0] for y in best_models[st][:3]])[[1,0,2]]

        # Plot null model adjustment for fitting a free variable
        ist, insamp = EM.get_weight_index(st, nsamp_max)
        if isinstance(nsamp_max, int):
            wsum = np.sum([weights[n][ist,insamp].sum() for n in EM.N_INTS])
        else:
            wsum = weightsO[ist,insamp].sum()

        if mode == 'A':
            null_2sigma = 2 / (wsum)**0.5
        elif mode == 'B':
            # Need to take into account fitting separate values of beta,
            # which depends on how many types of n_ints values are present in the set
            if isinstance(nsamp_max, int):
                n_beta_vals = np.sum([np.any(weights[n][ist,insamp] > 0) for n in EM.N_INTS])
            else:
                n_beta_vals = np.sum(np.any(weightsO[ist,insamp] > 0))
            null_2sigma = 2 * (n_beta_vals / wsum)**0.5

        ax.bar(X + xoff, Y, width, color=COL_ST[st], alpha=al, ec='k', lw=0.5)

        idx1 = null_2sigma < Y
        idx2 = (null_2sigma * 1.5) < Y
        if st == 'Theory':
            ax.plot(X[idx1] + xoff, Y[idx1] + 0.25, '*k')
            ax.plot(X[idx2] + xoff, Y[idx2] + 0.5, '*k')
        else:
            ax.plot(X[idx1] + xoff, Y[idx1] + 0.25, '*k')
            ax.plot(X[idx2] + xoff, Y[idx2] + 0.5, '*k')

    ax.set_xticks(X)
    ax.set_xticklabels(['OF', 'GP', 'HP'])
    ax.set_ylabel("Log-likelihood ratio per scale")
    ax.set_xlabel("Harmony composite models")
    handles = [Patch(ec='k', fc=COL_ST[st]) for st in SCALE_TYPE]
    ax.legend(handles, list(SCALE_TYPE), bbox_to_anchor=(1.25, 1.0), frameon=0, ncol=1)
    return ax






