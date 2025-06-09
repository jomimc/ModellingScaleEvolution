"""
This module contains miscellaneous functions.
"""
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.cluster import DBSCAN




###############################################################
### Scale features

def identify_scale_type(instrument):
    if isinstance(instrument, str):
        if instrument == '':
            return 'Theory'
        elif instrument == 'Voice':
            return 'Vocal'
        else:
            return 'Instrumental'
    return None


def scale_range(scale):
    return scale.max() - scale.min()


def int_min(ints):
    if len(ints):
        return min(ints)
    return np.nan


def int_max(ints):
    if len(ints):
        return max(ints)
    return np.nan


###############################################################
### Renaming Regions

def rename_regions(df):
    df = df.copy()
    regions = df.Region.unique()
    reg_key = {r:r for r in regions}
    reg_key.update({'Europe':'Eu', 'South America':'SA', 'Latin America':'SA', 'SA':'SA', 'Africa':'Af',
                    'Southeast Asia':'SEA', 'South East Asia':'SEA', 'South Asia':'SAs', 'CA':'CA',
                    'East Asia':'EA', 'EA':'EA', 'Oceania':'IP', 'North America':'NA', 'NA':'NA', 'Western':'Eu', 'Middle East':'ME'})
    df['Region'] = df.Region.map(reg_key)

    # Merging Insular Pacific and Australia into Oceania
    reg_key = {'IP':'Oc', 'Au':'Oc'}
    df['Region'] = df.Region.apply(lambda x: reg_key.get(x, x))

    # Fixing some incorrect assignments
    df.loc[df.Country=='Khmu', 'Region'] = 'SEA'
    df.loc[df.Country=='Indonesia', 'Region'] = 'SEA'

    return df


def rename_regions2(df):
    df = df.copy()
    regions = df.Region.unique()
    reg_key = {r:r for r in regions}
    reg_key.update({'Europe':'Eu', 'South America':'SAm', 'Latin America':'SAm', 'SA':'SAm', 'Africa':'Af',
                    'Southeast Asia':'SEAs', 'South East Asia':'SEAs', 'South Asia':'SAs', 'CA':'CAs',
                    'East Asia':'EAs', 'EA':'EAs', 'Oceania':'IP', 'North America':'NAm', 'NA':'NAm', 'Western':'Eu', 'Middle East':'ME'})
    df['Region'] = df.Region.map(reg_key)
    return df


def rename_countries(df):
    key = {}

    # EA
    key.update({x:'Japan' for x in ['Aomori', 'Ryukyu Okinawa', 'Ryukyu Yaeyama', 'Shizuoka', 'Tokushima', 'Japanese', 'Saga']})
    key.update({x:'South Korea' for x in ['Korea', 'Gangwon', 'South Jeolla']})
    key['Nivkh'] = 'Russia'

    # SEA
    key['Singapore'] = "Malaysia"
    ### This should be SEA, not EA
    key['Khmu'] = 'Laos'

    # SA
    key['Inca'] = 'Peru'
    key['Northern Brazil'] = 'Brazil'
    key['Saramaka'] = 'Suriname'
    key['Dominica'] = 'Dominican Rep.'

    # NA
    key.update({x:'United States of America' for x in ['USA', 'American Indians', 'Great Lake Indians', 'Kiowa-Apache', 'Navajo', 'Piegan', "Tohono O'Odham "]})
    key.update({x:'Canada' for x in ['Kwakiutl', 'Metis', 'Siksika']})

    # Ci
    key.update({f"{x} Greenland Inuit":'Greenland' for x in ['North', 'Eastern', 'South', 'West']})
    key.update({x:'Russia' for x in ['Jukaghir', 'Khanty', 'Koryak', 'Mansi', 'Nenec', 'Nganasan', 'Orok', 'Selkup', 'Yakut', 'Even', 'Evenk']})
    key['Saami'] = 'Finland'

    # CA
    key['Badakhshani/Pamiri'] = 'Tajikistan'
    key['Bukharan'] = 'Uzbekistan'
    key['Kara-kalpak'] = 'Uzbekistan'
    key['Khorezmian'] = 'Uzbekistan'

    # Af
    key['Central African Republic'] = 'Central African Rep.'
    key['Kalahari'] = 'Botswana'

    # Au
    key.update({x:'Australia' for x in ['Kokopera', 'Kunwinjku', 'North Australia', 'Nyangumarda', 'Warlpiri', 'Western Australia', 'Yindjibarndi', 'Yolngu']})

    # Eu
    key['Dutch'] = 'Netherlands'
    key['UK'] = 'United Kingdom'
    key['Bosnia-Herzegovina'] = 'Bosnia and Herz.'

    # IP
    key.update({x:'Taiwan' for x in ['Atayal', 'Boukan-hi', 'Kan-hi', 'Kan-kei', 'Kanakanavu', 'Kavalan', 'Tieu-hi', 'Truku; Formosa Aboriginal Song and Dance Troupe']})
    key['Ayan'] = 'Russia'
    
    # ME
    key['Bahrain'] = 'Qatar'
    key['Kurdistan'] = 'Iraq'
    key['Sinai Peninsula'] = 'Egypt'


    df['country_map'] = df.Country.apply(lambda x: key.get(x, x))

    return df


def add_latlong(df):
    data = [('Malta', 35.917973, 14.409943),
            ('French Guiana', 4, -53),
            ('Singapore', 1, 104),
            ('Nivkh', 53, 140),
            ('Ryukyu Okinawa', 26, 128),
            ('Ryukyu Yaeyama', 24, 124),
            ('American Indians', 40, -110),
            ('Great Lake Indians', 48, -125),
            ('Kiowa-Apache', 35, -98),
            ('Navajo', 36, -109),
            ('Piegan', 48, -113),
            ('Kwakiutl', 51, -127),
            ('Metis', 54, -117),
            ('Siksika', 51, -114),
            ("Tohono O'Odham", 31, -112),
            ('North Greenland Inuit', 79, -68),
            ('South Greenland Inuit', 68, -50),
            ('West Greenland Inuit', 71, -51),
            ('Eastern Greenland Inuit', 66, -37),
            ('Jukaghir', 71, 127),
            ('Khanty', 61, 69),
            ('Koryak', 61, 167),
            ('Mansi', 61, 69),
            ('Nenec', 66, 77),
            ('Nganasan', 71, 93),
            ('Orok', 51, 143),
            ('Selkup', 56, 85),
            ('Yakut', 67, 124),
            ('Even', 60, 151),
            ('Evenk', 54, 108),
            ('Saami', 69, 27),
            ('Kokopera', -11, 143),
            ('Kunwinjku', -12, 133),
            ('North Australia', -19, 133),
            ('Nyangumarda', -21, 122),
            ('Warlpiri', -19, 133),
            ('Western Australia', -28, 122),
            ('Yindjibarndi', -21, 117),
            ('Yolngu', -13, 135),
            ('Tonga', -21, 175),
            ('Marshall Islands', 7, 171),
            ('Bahrain', 26, 51),
            ('Kurdistan', 36, 44),
            ('Sinai Peninsula', 30, 34)]
    for c, x, y in data:
        df.loc[df.Country==c, ['latitude', 'longitude']] = [x, y]
    return df


###############################################################
### Mathematical functions

def nanmean_weight(X, W, axis):
    return np.nansum(X * W, axis=axis) / np.nansum(W)


def gini(prob):
#   prob = np.array(sorted(prob))
    prob = np.array([0] + sorted(prob))
    cumprob = np.cumsum(prob / prob.sum())
#   return 1 - np.trapz(cumprob, dx=1/prob.size) * 2
    diag = np.arange(1, prob.size + 1) / prob.size
    return np.sum((diag - cumprob)) / np.sum(diag)


def gini_seq(seq):
    return gini(list(Counter(seq).values()))


def sigmoid(X, a, b):
    return 1 / (1 + np.exp((X - a)/b))


def gaussian(X, mu, sigma):
    return np.exp(-((X - mu) / sigma)**2 / 2) / sigma / (2 * np.pi)**0.5


def gaussian_vector(X, mu, sigma):
    X = X.reshape(-1, 1)
    mu = mu.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    return gaussian(X, mu, sigma)


def get_kl_div_uniform(X):
    X = X / np.sum(X)
    return np.sum(X * np.log(X * X.size))


### Taken from stackoverflow
### https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
def conv_circ_fft( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))


### The fft version sometimes gives negative numbers due to
### insufficient precision at low numbers
def conv_circ( signal, ker ):
    return np.array([np.dot(signal, np.roll(ker, i)) for i in range(len(ker))])


###############################################################
### Complex Harmonic Tones


### interval :: interval between two frequencies, given in cents
### rho :: roll-off measured in dB / octave
def basic_timbre(interval, f1=440, n=10, rho=3):
    series = np.arange(1, n+1)

    if isinstance(interval, (float, int)):
        f2 = f1 * 2**(interval / 1200)
        partial1 = series * f1
        partial2 = series * f2
        weight = 10**(-rho * np.log2(series) / 20)

        partials = np.concatenate([partial1, partial2])
        weights = np.concatenate([weight, weight])
    elif isinstance(interval, (list, np.ndarray)):
        interval = np.array(interval)
        f2 = f1 * 2**(interval / 1200)
        partial1 = series.reshape(1,-1) * np.array([f1] * len(interval)).reshape(-1, 1)
        partial2 = series.reshape(1,-1) * f2.reshape(-1, 1)
        weight = np.array([10**(-rho * np.log2(series) / 20)] * len(interval))

        partials = np.concatenate([partial1, partial2], axis=1)
        weights = np.concatenate([weight, weight], axis=1)

    else:
        assert TypeError("Wrong 'interval' type passed to function. Needs to be one of (int, float) or (list, np.ndarray)")
    return partials, weights


def constrained_timbre(interval, f1=20, n=10, rho=3, limit=20000):
    partials, weights = basic_timbre(interval, f1=440, n=10, rho=3)
    print(np.sum(weights == 0))
    weights[partials > limit] = 0
    print(np.sum(weights == 0))
    return partials, weights


### interval :: interval between two frequencies, given in cents
### rho :: roll-off measured in dB / octave
def fancy_timbre(interval, series, f1=440, n=10, rho=3):
    if isinstance(interval, (float, int)):
        f2 = f1 * 2**(interval / 1200)
        partial1 = series * f1
        partial2 = series * f2
        weight = 10**(-rho * np.log2(series) / 20)

        partials = np.concatenate([partial1, partial2])
        weights = np.concatenate([weight, weight])
    elif isinstance(interval, (list, np.ndarray)):
        interval = np.array(interval)
        f2 = f1 * 2**(interval / 1200)
        partial1 = series.reshape(1,-1) * np.array([f1] * len(interval)).reshape(-1, 1)
        partial2 = series.reshape(1,-1) * f2.reshape(-1, 1)
        weight = np.array([10**(-rho * np.log2(series) / 20)] * len(interval))

        partials = np.concatenate([partial1, partial2], axis=1)
        weights = np.concatenate([weight, weight], axis=1)

    else:
        assert TypeError("Wrong 'interval' type passed to function. Needs to be one of (int, float) or (list, np.ndarray)")
    return partials, weights



###############################################################
### Harmonic overlap (should be moved to "cost_functions.py")


def entropy_gmm(means, variances, weights, npoints=10000):
    xmin, xmax = means.min(), means.max()
#   X = np.linspace(xmin - variances.max()*2, xmax + variances.max()*2, npoints)
    X = np.arange(xmin - variances.max()*5, xmax + variances.max()*5, 0.1)
    Y = np.average(gaussian_vector(X, means, variances), weights=weights, axis=1)
    plt.plot(X, Y)
    return entropy(Y)


def harmonic_overlap(partials, weights, sigma=6.83, dx=0.1):
    series = np.log2(partials / partials[0]) * 1200
    n = len(weights) // 2
    first_tone = (np.arange(n*2) < n).astype(bool)
    dx = sigma / 100
    extent = sigma * 3

    # Cluster means if they overlap 
    clust = DBSCAN(eps=extent, min_samples=1).fit(series.reshape(-1, 1)).labels_

    # Evaluate the error between terms in the harmonic series
    error = 0.0
    w_sum_tot = 0.0
    for c in np.unique(clust):
        idx = clust == c
        # If there is no overlap, then approximate the error as simply
        # the weight. Since there is no overlap within 3 sigma, we know the
        # lower bound on the error is 99.7% of the weight, so we are at worst
        # off by an amount of 0.3%.
        if np.sum(idx) == 1:
            error += weights[idx][0]
            continue

        xmin = np.min(series[idx])
        xmax = np.max(series[idx])
        X = np.arange(xmin - extent / 2, xmax + extent / 2, dx)
        gauss = [np.sum([weights[i] * gaussian(X, series[i], sigma) for i in np.where(idx & first_tone==j)[0]], axis=0) for j in [0,1]]
        error += np.sum(np.abs(gauss[0] - gauss[1])) * dx
    
    mae = (np.sum(weights) - error) / len(weights)
    return mae


###############################################################
### Correlations


def get_lr(X, Y, p=False, nozero=False, xlog=False, ylog=False, spear=False):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    try:
        if xlog:
            X = np.log(X)

        if ylog:
            Y = np.log(Y)

        if nozero:
            idx = (np.isfinite(X)) & (np.isfinite(Y)) & (X > 0) & (Y > 0)
        else:
            idx = (np.isfinite(X)) & (np.isfinite(Y))

        if spear:
            corr_fn = spearmanr
        else:
            corr_fn = pearsonr

        if p:
            return corr_fn(X[idx], Y[idx])
        else:
            return corr_fn(X[idx], Y[idx])[0]
    except Exception as e:
        print(e)
        if p:
            return [np.nan, np.nan]
        else:
            return np.nan


###############################################################
### Sampling functions


def sample_df_index(df, xsamp='Region', s=5):
    out = []
    for c in df[xsamp].unique():
        idx = df.loc[df[xsamp]==c].index
        out.extend(list(np.random.choice(idx, replace=False, size=min(s, len(idx)))))
    return np.array(out)


def sample_df_value(df, ysamp='Ints', xsamp='Region', s=5):
    out = []
    for c in df[xsamp].unique():
        Y = df.loc[df[xsamp]==c, ysamp]
        out.extend([x for y in np.random.choice(Y, replace=False, size=min(s, len(Y))) for x in y]) 
    return np.array(out)


def sample_df_value2(df, ysamp='Ints', xsamp='Region', s=5):
    out = []
    for c in df[xsamp].unique():
        Y = df.loc[df[xsamp]==c, ysamp]
        out.extend([x for x in np.random.choice(Y, replace=False, size=min(s, len(Y)))]) 
    return np.array(out)


def return_shuffled_scales(df):
    out = []
    for ints in df.Ints:
        np.random.shuffle(ints)
        out.append(ints)
    return out


def sample_shuffled_scales(df, xsamp='Region', s=5, inc_last=False):
    out = []
    for c in df[xsamp].unique():
        int_list = df.loc[df[xsamp]==c, 'Ints']
        for ints in np.random.choice(int_list, replace=False, size=min(s, len(int_list))):
            ints = np.array(ints)
            np.random.shuffle(ints)
            out.append(ints)
#           if inc_last:
#               out.extend(list(np.cumsum(ints)))
#           else:
#               out.extend(list(np.cumsum(ints[:-1])))
    return np.array(out)


def create_new_scales(df, n_rep=10):
    ints = [x for y in df.Ints for x in y]
    n_notes = df.Ints.apply(len).values
    df_list = []

    for i in range(n_rep):
        new_ints = [np.random.choice(ints, replace=True, size=n) for n in n_notes]
        new_df = df.copy()
        new_df.Ints = new_ints
        df_list.append(new_df)

    return df_list







