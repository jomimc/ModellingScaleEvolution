"""
This module, "prod_variances.py", contains code
to esimate the production variance (sigma**2) due to singing.

This requires downloading several datasets and reformatting them.
This data is not provided here, but one can figure out what was done
from following the references in the paper, and reading the code here.

The data can be found in the following locations:

Dagstuhl data:
    https://zenodo.org/records/4608395

Choral Singing Dataset (CSD):
    https://zenodo.org/records/1286570

Erkomaishvili dataset:
    https://zenodo.org/records/6900514

If you want quick answers about this, contact John McBride:
    jmmcbride@protonmail.com

"""
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

from scales_io import PATH_FIG

sys.path.insert(0, "/home/johnmcbride/projects/TranscriptionValidation/Src")
import process_raw_data
sys.path.insert(0, "/home/johnmcbride/projects/Scales/Elizabeth/StevenScales")
import analyse_scales


def erkomaishvilli():
    data = process_raw_data.process_erkomaishvili()
    X = np.concatenate([np.diff(d['note_cents']) for d in data])
    idx = np.isfinite(X)
    gm = GaussianMixture(6).fit(X.reshape(-1, 1)[idx])
    int_std = np.average(gm.covariances_[:,0,0]**0.5, weights=gm.weights_)
    print('ERK:', int_std)

### The number of notes in 'midi' is higher than in the transcriptions,
### but there are no new notes as far as I can tell, so it is useful
### for establishing the size of the scale.
def get_scale_variance_csd(T, C, on, off, note_cents, midi):
    N = np.unique(midi).size
    gm = GaussianMixture(N).fit(note_cents.reshape(-1, 1))
    bounds = list(np.convolve(sorted(gm.means_[:,0]), np.ones(2)/2, mode='valid')) + [note_cents.max() + 1]
    degree = np.digitize(note_cents, bounds)
    std = np.array([np.std(np.concatenate([C[(T>=on[i])&(T<=off[i])] for i in np.where(degree==j)[0]])) for j in range(N)])
    count = np.array([np.sum(degree==j) for j in range(N)])
    return np.average(std, weights=count)


def csd():
    data = process_raw_data.process_csd()
    X = np.concatenate([np.diff(d['note_cents']) for d in data])
    idx = np.isfinite(X) & (X > -780) & (X < 800)
    gm = GaussianMixture(13).fit(X.reshape(-1, 1)[idx])
    gm.means_.astype(int)
    int_std = np.average(gm.covariances_[:,0,0]**0.5, weights=gm.weights_)
    print('CSD int std:', int_std)

    scale_std = np.mean([get_scale_variance_csd(d['time'], d['cents'], d['note_on'], d['note_off'], d['note_cents'], d['midi']) for d in data])
    print('CSD scale std:', scale_std)
    

### 'midi' in this case correctly matches the transcriptions
def get_scale_variance_dag(T, C, on, off, midi):
    midi_uniq = np.unique(midi)
    std = np.array([np.std(np.concatenate([C[(T>=on[i])&(T<=off[i])] for i in np.where(midi==m)[0]])) for m in midi_uniq])
    count = np.array([np.sum(midi==m) for m in midi_uniq])
    return np.average(std, weights=count)


def dagstuhl():
    data = process_raw_data.process_dagstuhl()
    ints = defaultdict(list)
    cents = defaultdict(list)
    scale_std = []
    for d in data:
        T = d['time']
        F = d['freq']
        C = d['cents']
        on = d['note_on']
        off = d['note_off']
        note_freq = np.array([np.median(F[(T>=on[i])&(T<=off[i])]) for i in range(len(on))])

        cents1 = np.log2(note_freq / 440) * 1200
        cents0 = d['note_cents']

        ints1 = np.diff(cents1)
        ints0 = np.diff(cents0)
        for i0, i1 in zip(ints0, ints1):
            if i1 != 0.0:
                # Need to correct for octave errors here
                ints[i0].append(i1 - np.round((i1 - i0) / 1200) * 1200)

        scale_std.append(get_scale_variance_dag(T, C, on, off, cents0))

    I = sorted(list(ints.keys()))
    count = np.array([len(ints[i]) for i in I])
    std = np.array([np.nanstd(ints[i]) for i in I])
    idx = count > 2
    int_std = np.average(std[idx], weights=count[idx])
    print('DAG int std:', int_std)


    s_std = np.mean(scale_std)
    print('DAG scale std:', s_std)


if __name__ == "__main__":

    erkomaishvilli()
    csd()
    dagstuhl()



