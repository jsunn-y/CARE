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
import pandas as pd
import numpy as np
import random
from sciutil import SciUtil
from processing import *

u = SciUtil()
seed=42

def get_difference_level(predicted_ECs):
    """
    Calculates the difference level between two EC predictions.
    """
    counters = []

    for true_EC in predicted_ECs:
        #convert true_EC to a list
        if type(predicted_ECs) == str:
            predicted_ECs = [predicted_ECs]
        true_split = true_EC.split('.')

        for predicted in predicted_ECs:
            predicted_split = predicted.split('.')
            counter = 0
            for predicted, true in zip(predicted_split, true_split):
                if predicted == true:
                    counter += 1
                else:
                    break
            counters.append(4 - counter)
    return np.max(counters)

def make_cluster_split(df: pd.DataFrame, cluster_column_name: str, ec_column_name: str):
    """
    Take a sample of sequences with a certain clustering identity from the dataframe. 

    For a certain EC level take a single sample.
    """
    np.random.seed(seed)
    random.seed(seed)
    train_isolated = df[df['Duplicated clusterRes30'] == False]
    train_isolated = train_isolated[train_isolated['Duplicated EC'] == True]
    # Make a validation set that is completely held out.

    #sample a random one from each unique EC at level 3 for validation (i.e. not in training or the larger test set)
    validation = train_isolated.groupby('EC3').sample(1)
    u.dp([cluster_column_name, 'Training: ', len(train_isolated), 'Validation:', len(validation)])
    return validation

def make_promiscous_split(swissprot):
    # Make a validation set that is completely held out.
    np.random.seed(seed)
    random.seed(seed)
    promiscuous = swissprot[swissprot['Promiscuous']]
    # promiscuous = promiscuous[promiscuous['Duplicated clusterRes90'] == False] #prevent sequence identity from being too high, but then nothing comes out
    promiscuous = promiscuous.groupby(['Entry', 'Sequence']).agg({'EC number': lambda x: list(x)}).reset_index()
    promiscuous['Surprise Level'] = promiscuous['EC number'].apply(get_difference_level)
    promiscuous['Number of ECs'] = promiscuous['EC number'].apply(lambda x: len(x))
    # Check if there are duplicates in terms of EC and sequence
    promiscuous['Duplicated EC'] = promiscuous['EC number'].duplicated(keep=False)
    promiscuous['Duplicated Sequence'] = promiscuous['Sequence'].duplicated(keep=False)
    promiscuous = promiscuous.sort_values(['Duplicated EC', 'Surprise Level', 'Number of ECs'], ascending=False)
    # Keep ones which have duplicated ECs so that there is a homolog in the training dataaste
    promiscuous = promiscuous[promiscuous['Duplicated EC'] == True]
    # Make this so that we can just sample a single one
    promiscuous['EC number'] = [';'.join(sorted(ecs)) for ecs in promiscuous['EC number']]
    #sample a random one from each unique EC at level 3 for validation (i.e. not in training or the larger test set)
    promiscuous = promiscuous.groupby('EC number').sample(1)
    #promiscuous.drop_duplicates(subset='EC number', inplace=True)
    promiscuous.reset_index(inplace=True)
    promiscuous = promiscuous[promiscuous['Surprise Level'] >= 2]
    promiscuous = promiscuous[promiscuous['Duplicated EC'] == True]
    return promiscuous

def split(swissprot: pd.DataFrame, price_filepath: str, output_folder: str):
    """
    Generate the test and training splits for the dataset
    """
    swissprot['Duplicated clusterRes30'] = swissprot['clusterRes30'].duplicated(keep=False)
    swissprot['Duplicated clusterRes50'] = swissprot['clusterRes50'].duplicated(keep=False)
    swissprot['Duplicated EC'] = swissprot['EC number'].duplicated(keep=False)
    swissprot['Promiscuous'] = swissprot['Sequence'].duplicated(keep=False)
    not_promiscuous = swissprot[~swissprot['Promiscuous']]

    validation_30 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes30', 'Duplicated EC')
    validation_50 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes50', 'Duplicated EC')
    validation_70 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes70', 'Duplicated EC')
    validation_90 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes90', 'Duplicated EC')

    promiscuous = make_promiscous_split(swissprot)

    # Price is directly processed, the only thing we ensure is that the lengths are correct.
    price = pd.read_csv(price_filepath, sep='\t')
    #remove sequences in price that are in swissprot
    price = price[~price['Sequence'].isin(swissprot['Sequence'])]
    price['Length'] = price['Sequence'].apply(len)
    price = price[price['Length'] > 100]
    price = price[price['Length'] < 650]

    # Pool the test sequences
    test_pooled_seqs = pd.concat([validation_30, validation_50, validation_70, validation_90, price, promiscuous])['Sequence'].unique()
    #remove from the training set
    train_swissprot = swissprot[~swissprot['Sequence'].isin(test_pooled_seqs)]
    
    #save all of the generated splits
    train_swissprot.iloc[:,:-6].to_csv(f'{output_folder}protein_train.csv')
    validation_30.iloc[:,:-6].to_csv(f'{output_folder}30_protein_test.csv', index=False)
    validation_50.iloc[:,:-6].to_csv(f'{output_folder}30-50_protein_test.csv', index=False)
    validation_70.iloc[:,:-6].to_csv(f'{output_folder}50-70_protein_test.csv', index=False)
    validation_90.iloc[:,:-6].to_csv(f'{output_folder}70-90_protein_test.csv', index=False)
    price.to_csv(f'{output_folder}price_protein_test.csv', index=False)
    promiscuous.to_csv(f'{output_folder}promiscuous_protein_test.csv', index=False)

    #print the length of every saved file
    for split in [train_swissprot, validation_30, validation_50, validation_70, validation_90, price, promiscuous]:
        print(len(split))

    # Save all as fasta files as well
    make_fasta(validation_30, f'{output_folder}validation_30.fasta')
    make_fasta(validation_50, f'{output_folder}validation_50.fasta')
    make_fasta(validation_70, f'{output_folder}validation_70.fasta')
    make_fasta(validation_90, f'{output_folder}validation_90.fasta')
    make_fasta(price, f'{output_folder}price.fasta')
    make_fasta(promiscuous, f'{output_folder}promiscuous.fasta')

