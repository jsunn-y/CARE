#################################################################################
# Copyright (c) 2012-2024 Scott Chacon and others                               #
#                                                                               #    
# Permission is hereby granted, free of charge, to any person obtaining         #
# a copy of this software and associated documentation files (the               #
# "Software"), to deal in the Software without restriction, including           #
# without limitation the rights to use, copy, modify, merge, publish,           #
# distribute, sublicense, and/or sell copies of the Software, and to            #
# permit persons to whom the Software is furnished to do so, subject to         #
# the following conditions:                                                     #                                          
# The above copyright notice and this permission notice shall be                #
# included in all copies or substantial portions of the Software.               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,               #
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF            #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                         #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE        #  
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION        #
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION         #
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.               #
#################################################################################

"""
This is the main class that is used to analyse the results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, ttest_ind

import matplotlib as mpl
font = {'size' : 16}
mpl.rc('font', **font)
mpl.rc('lines', linewidth=1.5)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

#find the column labeled as ['0']
def find_column(df, label):
    for i, col in enumerate(df.columns):
        if col == label:
            return i
    return None


def get_accuracy_level(true_ECs, predicted_ECs):
    """
    based on a list of predicted_ECs, calculates the highest level of accuracy achieved
    """
    rows = []
    for i, pred_ec in enumerate(predicted_ECs):
        # i.e. not a none type
        if isinstance(true_ECs[i], str):
            true_ec = [ec.strip() for ec in true_ECs[i].split(';')]
            pred_ec = pred_ec.strip().split('.')
            # Since there may be multiple true ECs for a given EC (i.e. if it is promiscuous we want to ensure that we're 
            # Getting the "most accurate" one).
            best_mean_acc = 0
            best_acc = [0, 0, 0, 0]

            for ecs in true_ec:
                acc = [0, 0, 0, 0]
                for j, ec in enumerate(ecs.split('.')):
                    if pred_ec[j] == ec:
                        acc[j] = 1
                    else:
                        # Break once we stop getting the correct one
                        break
                if np.mean(acc) > best_mean_acc:
                    best_mean_acc = np.mean(acc)
                    best_acc = acc
        else:
            best_acc = [0, 0, 0, 0]
        rows.append(best_acc)
    return np.array(rows)

def calculate_k1_accuracy(test_df):
    """ Calculate the accuracy for each level for k=N for the EC dataset."""
    accs = get_accuracy_level(test_df['0'].values, test_df['EC number'].values)
    test_df['Level 1 acc'] = accs[:, 0]
    test_df['Level 2 acc'] = accs[:, 1]
    test_df['Level 3 acc'] = accs[:, 2]
    test_df['Level 4 acc'] = accs[:, 3]
    # Total acc
    return np.mean(accs[:, 0], accs[:, 1], accs[:, 2], accs[:, 3])

def predict_k_accuracy(test_df, k=10):
    """ For k results, pick the one which has the best accuracy."""
    accs = np.zeros((len(test_df), 4))
    for ec_number in range(0, k, 1):
        accs += get_accuracy_level(test_df[str(ec_number)].values, test_df['EC number'].values)
    # Binarise the accs
    accs[accs > 0] = 1 
    test_df['Level 1 acc'] = accs[:, 0]
    test_df['Level 2 acc'] = accs[:, 1]
    test_df['Level 3 acc'] = accs[:, 2]
    test_df['Level 4 acc'] = accs[:, 3]
    # Total acc
    return np.mean(test_df['Level 1 acc']), np.mean(test_df['Level 2 acc']), np.mean(test_df['Level 3 acc']), np.mean(test_df['Level 4 acc'])
