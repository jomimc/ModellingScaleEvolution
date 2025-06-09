from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns

PATH_FIG = Path('../Figures')

#######################################################
### Fit data from Zarate et al.


def load_zarate_acc():
    path_list = [Path(f'Perception/zarate_data/{a}_{b}.txt') for a in ["1a", "1b"] for b in ["mus", "nonmus"]]
    data = [np.loadtxt(p).T for p in path_list]
    names = [p.stem for p in path_list]
    return names, data


def load_zarate_dprime():
    path_list = [Path(f'Perception/zarate_data/2a_{b}.txt') for b in ["mus", "nonmus", "fixed", "roving"]]
    data = [np.loadtxt(p).T for p in path_list]
    names = [p.stem for p in path_list]
    return names, data


def get_sdt_accuracy(dI, sigma):
#   return 1 - norm.cdf(dI / 2, dI, sigma)
    return norm.cdf(dI / 2, 0, sigma)


def fit_fn(coef, X, Y, fn):
    Ypred = fn(X, *coef)
    return np.abs(Y - Ypred).sum()


def fit_accuracy(dI, acc):
    res = minimize(fit_fn, 10, (dI, acc, get_sdt_accuracy), bounds=[(1, 200)])
    return res.x



def fit_dprime(dI, dprime):
    return dI / dprime
    

def analyze_zarate():
    fig, ax = plt.subplots(1,2,figsize=(8,5))
    col = sns.color_palette()

    names, data = load_zarate_acc()
    for i, (X, Y) in enumerate(data):
        Y = Y / 100
        ax[0].plot(X, Y, 'o', c=col[i], label=names[i])

        sigma = fit_accuracy(X, Y)[0]
        Ypred = get_sdt_accuracy(X, sigma)
        ax[0].plot(X, Ypred, '-', c=col[i])
        print(sigma)

    names, data = load_zarate_dprime()
    for i, (X, Y) in enumerate(data):
        sigma = fit_dprime(X, Y)
        ax[1].plot(X, sigma, '-', c=col[i], label=names[i])

    for a in ax:
        a.legend(loc='best', frameon=False)
        

def zarate_paper_fig():
    fig, ax = plt.subplots(figsize=(8,5))
    col = sns.color_palette()

    names, data = load_zarate_acc()
    yi = np.arange(60, 75, 3)[::-1]
    yi = [72, 66, 69, 63]
    for i, (X, Y) in enumerate(data):
        Y = Y / 100
        ax.plot(X, Y * 100, 'o', c=col[i], label=names[i])

        sigma = fit_accuracy(X, Y)[0]
        Ypred = get_sdt_accuracy(X, sigma)
        ax.plot(X, Ypred * 100, '-', c=col[i])
        ax.text(140, yi[i], r"$\sigma_{per}$ = " + f"{int(round(sigma)):d} cents", c=col[i])
        print(sigma)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel(r"Interval difference, $\delta I$ / cents")
    ax.set_ylabel("Accuracy (% correct)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(PATH_FIG.joinpath("per_var_zarate.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("per_var_zarate.png"), bbox_inches='tight')

        

#######################################################
### Fit data from McDermott et al.


def jnd_vs_sigma(acc=0.707):
    dI = np.arange(1, 251, 5)
    sigma = []
    for di in dI:
        res = minimize(fit_fn, di, (np.array([di]), np.array([acc]), get_sdt_accuracy), bounds=[(1, 500)])
        sigma.append(res.x[0])
    fig, ax = plt.subplots(1,2,figsize=(8,5))
    ax[0].plot(dI, sigma)
    ax[0].plot(dI, dI, '-k')

    jnd = np.loadtxt('Perception/mcdermott_data/fig2_PC_ID.txt')[:,1].reshape(-1,3)
    jnd_mean = jnd.mean(axis=1)
    labels = ['NM'] * 5 + ['AM'] * 3 + ['PM'] * 3
    
    jnd_fit = [sigma[np.argmin(np.abs(j - dI/100))] for j in jnd_mean]
    sns.histplot(jnd_fit, ax=ax[1], binwidth=10)
    


#######################################################
### Fit data from Scherbaum et al.


def erkomaishvilli_mel_int_std():
    X, Y = np.loadtxt('Intonation/scherbaum_data/mel_int_hist.txt').T
    xgrid = np.arange(2.5, 250, 5)
    vals = []
    for x, y in zip(X, Y):
        count = int(round(y))
        vals.extend([xgrid[np.argmin(np.abs(xgrid - x))]]*count)

    print(np.mean(vals))
    print(np.std(vals))


#######################################################
### Fit data from Pfordresher and Brown


def pfordresher_and_brown():
    semitone_deviation = np.loadtxt('Intonation/pfordresher_data/fig4a.txt')[:,0]
    sigma = np.arange(10, 410, 5)
    sd = np.array([np.std(np.abs(norm.rvs(0, s, size=100000))) for s in sigma])
    fig, ax = plt.subplots(1,2,figsize=(8,5))
    ax[0].plot(sigma, sd)

    sd_fit = [sigma[np.argmin(np.abs(sd - s*100))] for s in semitone_deviation]
    sns.histplot(sd_fit, ax=ax[1])
    print(np.min(sd_fit), np.max(sd_fit))




