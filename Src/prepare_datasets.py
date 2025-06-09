"""
This module, "prepare_datasets.py", contains code for amalgamating the three separate
sources of scales used in this work.

This will not work 100% for others, as it requires access to some files from other
projects that were not fully released. In any case, all the data (input, and output)
is provided in this Github repository in the "Data" folder. This code is mainly provided
for transparency, to show how the three datasets were merged.


Most of the raw data can be obtained at:
    https://zenodo.org/records/10608046
    https://zenodo.org/record/7250281


See the bottom of the code for details of the main functions to run,
in order to reproduce the work.

"""
import sys
sys.path.insert(0, "/home/johnmcbride/projects/Scales/Elizabeth/StevenScales")
sys.path.insert(0, "/home/johnmcbride/projects/BirdSongSpeech/Data/HumanMusic/Savage_et_al_2015")

import numpy as np
import pandas as pd

import analyse_scales
import collate_data

def process_steven():
    analyse_scales.save_reduced_dataset(analyse_scales.load_scales())


def process_garland():
    collate_data.collate_data()


def process_damusc():
    df1 = pd.read_csv('/home/johnmcbride/projects/Scales/DaMuSc/Data/measured_scales.csv')

    # Removing a scale that I think is incorrect due to a typo in print
    df1 = df1.loc[df1.MeasuredID!="M0397"]

    df1 = df1.rename(columns={'MeasuredID':'ScaleID',
                     'Measured_type':'Method', 'Intervals':'Ints'})
    df1['Instrument'] = df1['Inst_type']

    col = ['ScaleID', 'Instrument', 'Country', 'Region', 'Tonic', 'Method', "Ints", "SocID", "Name"]
    df1 = df1.loc[:, col]

    # Removing Instrument scales where we don't know the Instrument 
    df1 = df1.loc[df1.Instrument.notnull()]

    # Fixing Instrument type for one reference
    df1.loc[df1.Instrument=='Varied', 'Instrument'] = 'Chordophone'
    
    
    df2 = pd.read_csv('/home/johnmcbride/projects/Scales/DaMuSc/Data/octave_scales.csv')
    df2 = df2.loc[df2.Theory=='Y']
    df2 = df2.rename(columns={'step_intervals':'Ints'})
    df2['Method'] = 'Theory'
    df2['Instrument'] = ''
    df2['Tonic'] = 0
    df2 = df2.loc[:, col]

    df = pd.concat([df1, df2], ignore_index=True)
    df['Collection'] = 'DaMuSc'

    df['Ints'] = df.Ints.apply(lambda x: np.array(x.split(';'), int))
    df['Scale'] = df.Ints.apply(lambda x: np.cumsum([0] + list(x)))
    print(df.columns)

    df.to_pickle('/home/johnmcbride/projects/Scales/Evolution/Data/dataset_damusc.pkl')


def prepare_all():
    process_steven()
    process_garland()
    process_damusc()


if __name__ == "__main__":

    prepare_all()


