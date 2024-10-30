import unittest

import pandas as pd


base_dir = ''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, ttest_ind

import matplotlib as mpl

font = {'size': 16}
mpl.rc('font', **font)
mpl.rc('lines', linewidth=1.5)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2


# to the column labeled as ['0']
def find_column(df, label):
    """
    finds the column index of a given label
    """
    for i, col in enumerate(df.columns):
        if col == label:
            return i
    return None


def get_average_accuracy(accuracy_level_list, level):
    """
    returns the average accuracy at a given level
    """
    return np.mean([1 if x >= level else 0 for x in accuracy_level_list])


def expand_list(predicted_ECs, on="; "):
    """
    for every entry in predicted_ECs, split by ';' and expand the list
    """
    expanded = []
    for entry in predicted_ECs:
        # if entry is a list
        if type(entry) == list:
            return entry
        else:
            expanded.extend(entry.split(on))
    return expanded


def get_accuracy_level(predicted_ECs, true_ECs):
    """
    based on a list of predicted_ECs, calculates the highest level of accuracy achieved, against all true_ECs. Returns a list of the same length as true_ECs.
    """
    # convert true_EC to a list
    if type(predicted_ECs) == str:
        predicted_ECs = [predicted_ECs]

    if type(true_ECs) == str:
        true_ECs = [true_ECs]

    maxes = []
    err_c = 0

    for true_EC in true_ECs:

        true_split = true_EC.split('.')

        counters = []
        for predicted_EC in predicted_ECs:
            try:
                # if the output is not an EC number
                if predicted_EC.count('.') != 3:
                    err_c += 1
                    predicted_EC = '0.0.0.0'

                predicted_split = predicted_EC.split('.')
                counter = 0

                for predicted, true in zip(predicted_split, true_split):
                    if predicted == true:
                        counter += 1
                    else:
                        break
                counters.append(counter)
            except:
                print("ERROR:", predicted_EC)

        maxes.append(np.max(counters))
    return maxes


def shorten_list(predicted_ECs, k):
    predicted_ECs = predicted_ECs[:k]
    return predicted_ECs


def get_results(task):
    results = pd.DataFrame(
        columns=['baseline', 'split', 'k', 'level 4 accuracy', 'level 3 accuracy', 'level 2 accuracy',
                 'level 1 accuracy'])

    # load the query_df that's already been generated

    if task == 'task1':
        # baselines = ['random', 'BLAST', 'ChatGPT', 'ProteInfer',  'CLEAN']
        baselines = ['ChatGPT', 'BLAST'] #'random', 'BLAST', 'CLEAN', 'Pika', 'ChatGPT']  # 'random', 'BLAST', 'CLEAN', 'Pika', 'ChatGPT'
        splits = ['30', '30-50', 'price', 'promiscuous']  # ['30', '30-50', 'price', 'promiscuous']
        modality = 'protein'
    else:
        # baselines = ['random', 'Similarity', 'ChatGPT_text', 'ChatGPT', 'CLIPZyme', 'CREEP', 'CREEP_text']
        baselines = ['ChatGPT'] #['random', 'Similarity', 'CLIPZyme', 'CREEP', 'ChatGPT_text', 'ChatGPT', 'CREEP_text']
        splits = ['easy', 'medium', 'hard']
        modality = 'reaction'

    for baseline in baselines:
        for split in splits:

            query_df = pd.read_csv(
                '{}_baselines/results_summary/{}/{}_{}_test_results_df.csv'.format(task, baseline, split,
                                                                                   modality))  # take a different baseline and randomize it
            num_cols = find_column(query_df, '0')

            # fill na for columns after num_cols with '0.0.0.0'
            query_df.iloc[:, num_cols:] = query_df.iloc[:, num_cols:].fillna('0.0.0.0')

            # print(query_df)

            # change this line if you have fewer rankings available
            if baseline == 'Pika' or baseline == 'BLAST' or "ChatGPT" in baseline or baseline == 'test':
                k_list = [1]
            elif task == 'task1':
                if baseline == 'CLEAN' or baseline == 'random':
                    k_list = [1, 20]  # more than the maximum number of promiscuous ECs
            else:
                k_list = [1, 3, 5, 10, 20, 30, 40, 50]

            for k in k_list:
                # collapse columns 0:3 into a single column list
                query_df['predicted ECs'] = query_df.iloc[:, num_cols:num_cols + k].values.tolist()
                # for each entry in the list, split out entries that contain ';'
                if baseline == 'BLAST':
                    query_df['predicted ECs'] = query_df['predicted ECs'].apply(lambda x: expand_list(x))
                elif baseline == 'Pika' or "ChatGPT" in baseline:
                    query_df['predicted ECs'] = query_df['predicted ECs'].apply(lambda x: expand_list(x, on=','))
                elif baseline == 'CLEAN' or baseline == 'random':
                    # shorten listen to the number of entries in "EC number", based on the value of "Number of ECs"
                    query_df['Number of ECs'] = query_df['EC number'].apply(lambda x: x.count(';') + 1)
                    query_df['predicted ECs'] = query_df.apply(
                        lambda x: shorten_list(x['predicted ECs'], x['Number of ECs']), axis=1)

                query_df['EC number list'] = query_df['EC number'].apply(lambda x: x.split(';'))

                query_df['k={} accuracy level'.format(k)] = query_df.apply(
                    lambda x: get_accuracy_level(x['predicted ECs'], x['EC number list']), axis=1)

                accuracies = []
                for i in [4, 3, 2, 1]:
                    query_df['accuracy'] = query_df['k={} accuracy level'.format(k)].apply(
                        lambda x: get_average_accuracy(x, i))
                    # if baseline == 'CLEAN' and split == 'promiscuous':
                    #     print(query_df[['EC number', 'predicted ECs', 'accuracy']])
                    accuracy = query_df['accuracy'].mean()

                    accuracies.append(round(accuracy * 100, 1))

                results.loc[len(results)] = [baseline, split, k, accuracies[0], accuracies[1], accuracies[2],
                                             accuracies[3]]

    return results


class TestAccuracy(unittest.TestCase):

    def test_k1_acc(self):
        """ Test the accuracy of k=1 for task 1 """
        results = get_results('task1')
        results['baseline'] = pd.Categorical(results['baseline'], ['ChatGPT', 'BLAST'])
        results.to_csv('test_accuracy.csv', index=False)

    def test_task2_acc(self):
        """ Test the accuracy of k=1 for task 1 """
        results = get_results('task2')
        results['baseline'] = pd.Categorical(results['baseline'], ['ChatGPT'])
        results.to_csv('test_accuracy_task2.csv', index=False)

if __name__ == '__main__':
    unittest.main()
