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

def get_average_accuracy(accuracy_level_list, level):
    """
    returns the average accuracy at a given level
    """
    return np.mean([1 if x >= level else 0 for x in accuracy_level_list])

def expand_list(predicted_ECs):
    """
    for every entry in predicted_ECs, split by ';' and expand the list
    """
    expanded = []
    for entry in predicted_ECs:
        # if entry is a list
        if type(entry) == list:
            print('ERROR: ')
            print(entry)
            return entry
        else:
            expanded.extend(entry.split(';'))
    return expanded

def get_accuracy_level(predicted_ECs, true_ECs):
    """
    based on a list of predicted_ECs, calculates the highest level of accuracy achieved, against all true_ECs. Returns a list of the same length as true_ECs.
    """
    #convert true_EC to a list
    if type(predicted_ECs) == str:
        predicted_ECs = [predicted_ECs]
        
    if type(true_ECs) == str:
        true_ECs = [true_ECs]

    maxes = []
    for true_EC in true_ECs:

        true_split = true_EC.split('.')
        
        counters = []
        for predicted_EC in predicted_ECs:
            try:
                predicted_split = predicted_EC.split('.')
                counter = 0
    
                for predicted, true in zip(predicted_split, true_split):
                    if predicted == true:
                        counter += 1
                    else:
                        break
                counters.append(counter)
                #print(counters)
            except:
                print("ERROR:", predicted_EC)
        
        maxes.append(np.max(counters))
    return maxes

def get_accuracy_level_v2(true_ECs, predicted_ECs):
    """
    based on a list of predicted_ECs, calculates the highest level of accuracy achieved
    """
    rows = []
    for i, pred_ecs in enumerate(predicted_ECs):
        # Since for promisc when we get it from blast it may contain ; since they are just matching, we need to check all 
        best_mean_acc = 0
        best_acc = [0, 0, 0, 0]
        for pred_ec in pred_ecs.split(';'):
            # i.e. not a none type
            if isinstance(true_ECs[i], str):
                true_ec = [ec.strip() for ec in true_ECs[i].split(';')]
                pred_ec = pred_ec.strip().split('.')
                # Since there may be multiple true ECs for a given EC (i.e. if it is promiscuous we want to ensure that we're 
                # Getting the "most accurate" one).
                
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
        rows.append(best_acc)
    return np.array(rows)

def predict_k_accuracy(test_df, k=10):
    """ For k results, pick the one which has the best accuracy."""
    accs = np.zeros((len(test_df), 4))
    for ec_number in range(0, k, 1):
        print(ec_number)
        try:
            accs += get_accuracy_level_v2(test_df[str(ec_number)].values, test_df['EC number'].values)
        except:
            print("Not that many ECs in dataset.")
    # Binarise the accs
    accs[accs > 0] = 1 
    test_df['Level 1 acc'] = accs[:, 0]
    test_df['Level 2 acc'] = accs[:, 1]
    test_df['Level 3 acc'] = accs[:, 2]
    test_df['Level 4 acc'] = accs[:, 3]
    # Total acc
    return [np.mean(test_df['Level 4 acc']), np.mean(test_df['Level 3 acc']), np.mean(test_df['Level 2 acc']), np.mean(test_df['Level 1 acc'])]

def get_k_acc(df: pd.DataFrame, list_of_ks: list, rows: list, tool_name: str, dataset_split: str):
    """
    Get K accuracy across a range of K's and add it to the rows that are passed
    """
    for k in list_of_ks:
        k_acc = predict_k_accuracy(df, k)
        rows.append([tool_name, dataset_split, str(k)] + k_acc)
        print(f'K{k} accuracy: ', k_acc)
    return rows