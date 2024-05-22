from __future__ import print_function
import mdtraj as mdt
import numpy as np
import collections
from collections import defaultdict
import pandas as pd
from contact_map import ContactMap, ContactFrequency, ContactDifference, ResidueContactConcurrence, plot_concurrence
import pickle
import warnings

warnings.filterwarnings("ignore", message="top= kwargs ignored since this file parser does not support it")

# Input parameters. Needed to find and name the files.

Protein = 'DNMT3A'      # Name of the protein. 
Variant = 'WT'          # WT: Wild Type or MT: Mutant.
sim_time = '25ns'       # simulation time of each replicate. usually something like: '100ns' 
replicates = 1          # number of replciates one wants to analyse in the current folder 
contact_cutoff = '0_35' # size of the sphere used to calculate the contacts in.


# want to subtract two DataFrames? - DataFrames must have same length.

subtract = False
substract_target = 'MT'
replicates_subtract_target = '1'

# load pickle file in which contact frequencies are stored. Generated from "DNMT3A-Contact_Maps.py".

infile = open('DNMT3A-{}-{}x{}_cutoff_{}.pkl'.format(Variant, replicates, sim_time, contact_cutoff), 'rb')
df = pickle.load(infile)
infile.close()
print('pickle loded')

if subtract == True:
    # load DataFrame that should be subtracted and subtract the two DataFrames.
    infile = open('DNMT3A-{}-{}x{}.pkl'.format(substract_target, replicates_subtract_target, sim_time), 'rb')
    df_subt = pickle.load(infile)
    infile.close()
    df = df - df_subt.values
    print('subtracting:{}-{}...'.format(Variant, substract_target))

# replace all values in DataFrame in the range of (-0.5) - 0.5 with NaN

df_cut = df.where((df >= 0.05) | (df <= -0.05), np.nan)

# delete all rows with only NaN in it (remaining NaN will not appear in excel)

df_drop = df_cut.dropna(axis=0, how='all')
df_drop2 = df_drop.dropna(axis=1, how='all')

# convert dataframe to excel

if subtract == True:
    print('given xlsx is:{}-{}'.format(Variant, substract_target))
    df_drop2.to_excel('contacts_{}_{}-{}_cutoff_{}.xlsx'.format(Protein, Variant, substract_target, contact_cutoff), index_label='{}-{}'.format(Variant, substract_target))
if subtract == False:
    print('given xlsx is:{}'.format(Variant))
    df_drop2.to_excel('contacts_{}_{}_{}x{}_cutoff_{}.xlsx'.format(Protein, Variant, replicates, sim_time, contact_cutoff), index_label='{}'.format(Variant))
