"""
This module contains global variables.
"""
from pathlib import Path
import platform

import numpy as np
import pandas as pd


if platform.node() == "johnmcbride-X399-DESIGNARE-EX":
    PATH_BASE = Path("/home/johnmcbride/projects/Scales/Evolution")
    N_PROC = 20

elif platform.node() == "bigram1":
    PATH_BASE = Path("/home/jmcbride/Scales/Evolution")
    N_PROC = 60

elif platform.node() == "bigram2":
    PATH_BASE = Path("/home/jmcbride/Scales/Evolution")
    N_PROC = 80

elif platform.node() == "login":
    PATH_BASE = Path("/home/jmcbride/Scales/Evolution")
    N_PROC = 28

PATH_DATA = PATH_BASE.joinpath("Data")
PATH_RES  = PATH_BASE.joinpath("Results")
PATH_COMP = PATH_BASE.joinpath("Precompute")
PATH_FIG = PATH_BASE.joinpath("Figures")
PATH_FIG_DATA = PATH_FIG.joinpath("Data")


MODEL_COL   = np.array(["#E06639", "#E06DB9", "#DB3B3B", "#6EB7DF", "#5A63E0"])
SCALE_TYPE  = np.array(['Vocal', 'Instrumental', 'Theory'])
MAIN_MODELS = np.array(['H_GP', 'H_GFO', 'H_HP', 'C_NI'])
MODEL_NAMES = np.array(['HI-ALL', 'HI-OF', 'HI-VAR', r'SC'])

HI_MODELS = np.array(['H_GP', 'H_GFO', 'H_HP', 'I_HK', 'I_SE', 'I_BE'])
HI_NAMES = np.array(['H-GP', 'H-OF', 'H-HP', 'I-HK', 'I-S', 'I-B'])

REGIONS = ['NA', 'SA', 'Eu', 'Af', 'ME', 'CA', 'EA', 'SAs', 'SEA', 'Ci', 'Oc']
REG_FULL = ['North America', 'South America', 'Europe', 'Africa',
            'Middle East', 'Central Asia', 'East Asia', 'South Asia',
            'Southeast Asia', 'Circumpolar', 'Oceania']



