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
import npysearch as npy
from openai import OpenAI


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

def make_cluster_split(df: pd.DataFrame, cluster_column_name: str, ec_column_name: str, entries_to_omit = None):
    """
    Take a sample of sequences with a certain clustering identity from the dataframe. 

    For a certain EC level take a single sample.
    """
    np.random.seed(seed)
    random.seed(seed)
    train_isolated = df[df[cluster_column_name] == False]
    train_isolated = train_isolated[train_isolated[ec_column_name] == True]
    # Make a validation set that is completely held out.
    # If we have entries to omit remove those (i.e. clusters from the previous ones)
    if entries_to_omit is not None:
        train_isolated = train_isolated[~train_isolated['Entry'].isin(entries_to_omit)]
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
    # First save the entirety of the protein file to the output folder as they will use this to map EC numbers
    swissprot.to_csv(os.path.join(output_folder, 'protein2EC.csv'), index=False)

    # Then check which ECs have dups.
    swissprot['Duplicated clusterRes30'] = swissprot['clusterRes30'].duplicated(keep=False)
    swissprot['Duplicated clusterRes50'] = swissprot['clusterRes50'].duplicated(keep=False)
    swissprot['Duplicated clusterRes70'] = swissprot['clusterRes70'].duplicated(keep=False)
    swissprot['Duplicated clusterRes90'] = swissprot['clusterRes90'].duplicated(keep=False)

    swissprot['Duplicated EC'] = swissprot['EC number'].duplicated(keep=False)
    swissprot['Promiscuous'] = swissprot['Sequence'].duplicated(keep=False)
    not_promiscuous = swissprot[~swissprot['Promiscuous']]

    validation_30 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes30', 'Duplicated EC')
    entries_to_omit = list(validation_30['Entry'].values)
    validation_50 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes50', 'Duplicated EC', entries_to_omit)
    entries_to_omit += list(validation_50['Entry'].values)
    validation_70 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes70', 'Duplicated EC', entries_to_omit)
    entries_to_omit += list(validation_70['Entry'].values)
    validation_90 = make_cluster_split(not_promiscuous, 'Duplicated clusterRes90', 'Duplicated EC', entries_to_omit)

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
    make_fasta(train_swissprot, f'{output_folder}protein_train.fasta')
    make_fasta(validation_30, f'{output_folder}30_protein_test.fasta')
    make_fasta(validation_50, f'{output_folder}30-50_protein_test.fasta')
    make_fasta(validation_70, f'{output_folder}50-70_protein_test.fasta')
    make_fasta(validation_90, f'{output_folder}70-90_protein_test.fasta')
    make_fasta(price, f'{output_folder}price_protein_test.fasta')
    make_fasta(promiscuous, f'{output_folder}promiscuous_protein_test.fasta')

# ----------------------------------------------------------------------------------
#                   Class that calls each tool.
# ----------------------------------------------------------------------------------
class Task1:

    def __init__(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder

    def get_train_fasta(self):
        return os.path.join(self.data_folder, 'protein_train.fasta')
    
    def get_train_df(self):
        return pd.read_csv(os.path.join(self.data_folder, f'protein_train.csv'))
    
    def get_test_df(self, label):
        return pd.read_csv(os.path.join(self.data_folder, f'{label}_protein_test.csv'))

    def get_test_fasta(self, label: str):
        if label not in ['train', '30', '30-50', 'price', 'promiscuous']:
            print(f'{label} not a valid dataset select one of ' + ' '.join(['30', '30-50', 'price', 'promiscuous']))
            return None
        else:
            print(os.path.join(self.data_folder, f'{label}_protein_test.fasta'))
            return os.path.join(self.data_folder, f'{label}_protein_test.fasta')

    def get_uniprot2ec(self):
        df = pd.read_csv(os.path.join(self.data_folder, 'protein2EC.csv'))
        return dict(zip(df['Entry'], df['EC number']))
    
    def get_price2ec(self):
        df = pd.read_csv(os.path.join(self.data_folder, 'price_protein_test.csv'))
        return dict(zip(df['Entry'], df['EC number']))

    def get_ChatGPT(self, test_label, n=10, save=False, api_key=None, subsample=None):
        """
        Gets the results for a series of ECs and formats it correctly for the paper
        """
        client = OpenAI(api_key=api_key)
        df = self.get_test_df(test_label)
        if subsample is not None: # Just for testing so we don't run too many
            df = df.sample(subsample)
        rows = []
        for entry, true_ec, seq in df[['Entry', 'EC number', 'Sequence']].values:
            text = f"Return the top {n} most likely EC numbers as a comma separated list for this enzyme sequence: {seq}"

            completion = client.chat.completions.create(
                model='gpt-4',
                messages=[
                    {"role": "system",
                    "content": 
                    "You are protein engineer capable of predicting EC numbers from a protein seqeunce alone."
                    + "You are also a skilled programmer and able to execute the code necessary to predict an EC number when you can't use reason alone." 
                    + "Given a protein sequence you are able to determine the most likely enzyme class for a seqeunce." 
                    + "You don't give up when faced with a sequence you don't know, you will use tools to resolve the most likely enzyme sequence."
                    + "You only return enzyme commission numbers in a comma separated list, no other text is returned, you have failed if you do "
                    + " not return the EC numbers. You only return the exact number of EC numbers that a user has provided requested, ordered by their likelihood of being correct."},
                    {"role": "user", "content": text}
                ]
            )
            preds = completion.choices[0].message.content.replace(" ", "").split(',')
            for p in preds:
                rows.append([entry, true_ec, p, seq]) # Costs you ~1c per query
        results = pd.DataFrame(rows)
        results.columns = ['entry',  'true_ecs', 'predicted_ecs', 'seq']
        grped = results.groupby('entry')
        max_ecs = 0
        rows = []
        for query, grp in grped:
            # Always will be the same for the grouped 
            true_ec = ';'.join(set([c for c in grp['true_ecs'].values]))
            seq = grp['seq'].values[0] # Only returns one sequence when there could be multiple
            # Filter to only include rows which were not null
            grp = grp[~grp['predicted_ecs'].isna()]
            grp = grp[grp['predicted_ecs'] != 'None']
            grp = grp.sort_values(by='predicted_ecs', ascending=False)

            if len(list(grp['predicted_ecs'].values)) > max_ecs:
                max_ecs = len(list(grp['predicted_ecs'].values))
            if len(list(grp['predicted_ecs'].values)) == 0:
                rows.append([query, true_ec, seq, ''])
            else:
                rows.append([query, true_ec, seq] + list(grp['predicted_ecs'].values))
        new_df = pd.DataFrame(rows)
        new_df.columns = ['Entry', 'EC number', 'Sequence'] + list(range(0, max_ecs))

        # Save to a file in the default location
        if save:
            new_df.to_csv(os.path.join(self.output_folder, f'ChatGPT_{test_label}_protein_test_results_df.csv'), index=False)
        return new_df
    
    
    def get_blast(self, test_label, num_ecs=1, min_identity=0.1, save=False):
        """
        Gets the results for blast for a series of ECs and formats it correctly for the paper
        """
        # Lets also look at the protein our query is the query genome and our database is going to be ecoli.
        
        results_prot = npy.blast(query=self.get_test_fasta(test_label),
                                database=self.get_train_fasta(),
                                minIdentity=min_identity,
                                maxAccepts=num_ecs,
                                alphabet="protein")
        results = pd.DataFrame(results_prot)  # Convert this into a dataframe so that we can see it more easily
        uniprot_to_ec = self.get_uniprot2ec()
        results['predicted_ecs'] = results['TargetId'].map(uniprot_to_ec)
        if test_label == 'price':
            results['true_ecs'] = results['QueryId'].map(self.get_price2ec())
        else:
            results['true_ecs'] = results['QueryId'].map(uniprot_to_ec)
        grped = results.groupby('QueryId')
        rows = []
        # Get the raw test set and then build the new dataset based on that! 
        # Get the raw test set and then build the new dataset based on that! 
        test_df = self.get_test_df(test_label)
        # Now we want to iterate through and get the predicted EC numbers
        entry_to_ec = dict(zip(test_df['Entry'], test_df['EC number']))
        entry_to_seq = dict(zip(test_df['Entry'], test_df['Sequence']))

        for query in test_df['Entry'].values:
            try:
                grp = grped.get_group(query)
                # Get all the ECs for all the seqs and join them!
                true_ec = ';'.join(set([uniprot_to_ec.get(target) for target in grp['TargetId'].values]))
                targets = ';'.join([ec for ec in grp['TargetId'].values]) # Also keep track of these just incase
                # Keep only the closest sequence 
                rows.append([query, targets, true_ec, grp['QueryMatchSeq'].values[0]] + list(grp['predicted_ecs'].values))
            except:
                u.warn_p([query, f'Had no sequences within {min_identity}.'])
                rows.append([query, '', entry_to_ec[query], entry_to_seq[query]])

        new_df = pd.DataFrame(rows)
        new_df.columns = ['Entry', 'Similar Enzymes', 'EC number', 'Sequence'] + list(range(0, num_ecs))
        
        # Save to a file in the default location
        if save:
            new_df.to_csv(os.path.join(self.output_folder, f'BLAST_{test_label}_protein_test_results_df.csv'), index=False)
        return new_df
    

    def get_proteinfer(self, test_label, proteinfer_dir: str, save=False):
        """
        Gets the results for a series of ECs and formats it correctly for the paper
        """
        # Run proteInfer NOTE! Must have ProteInfer environment already made
        u.dp(['Warning! You must already have the proteInfer environment made'])

        output_file = os.path.join(self.output_folder, test_label + "_proteInfer.tsv")
        cwd = os.getcwd()
        # Change to proteinfer dir to execute it
        os.chdir(proteinfer_dir)
        os.system(f'conda run -n proteinfer python3 {proteinfer_dir}proteinfer.py -i {self.get_test_fasta(test_label)} -o {output_file}')
        # Change back to cwd
        os.chdir(cwd)

        results = pd.read_csv(output_file, sep='\t')
        results['predicted_ecs'] = [ec.split(':')[1] if 'EC:' in ec else 'None' for ec in results['predicted_label'].values]
        if test_label == 'price':
            results['true_ecs'] = results['sequence_name'].map(self.get_price2ec())
        else:
            results['true_ecs'] = results['sequence_name'].map(self.get_uniprot2ec())

        grped = results.groupby('sequence_name')
        max_ecs = 0
        rows = []
        # Get the raw test set and then build the new dataset based on that! 
        test_df = self.get_test_df(test_label)
        # Now we want to iterate through and get the predicted EC numbers
        entry_to_ec = dict(zip(test_df['Entry'], test_df['EC number']))
        entry_to_seq = dict(zip(test_df['Entry'], test_df['Sequence']))

        for query in test_df['Entry'].values:
            if query not in results['sequence_name'].values:
                rows.append([query, entry_to_ec.get(query), entry_to_seq.get(query), None])
            else:
                grp = grped.get_group(query)
                # Get all possible ECs
                true_ec = ';'.join(set([ec for ec in grp['true_ecs'].values]))
                # Filter to only include rows which were not null
                grp = grp[~grp['predicted_ecs'].isna()]
                grp = grp[grp['predicted_ecs'] != 'None']
                grp = grp.sort_values(by='predicted_ecs', ascending=False)

                if len(list(grp['predicted_ecs'].values)) > max_ecs:
                    max_ecs = len(list(grp['predicted_ecs'].values))
                if len(list(grp['predicted_ecs'].values)) == 0:
                    rows.append([query, true_ec, entry_to_seq.get(query), None])
                else:
                    rows.append([query, entry_to_ec.get(query), entry_to_seq.get(query)] + list(grp['predicted_ecs'].values))

        new_df = pd.DataFrame(rows)
        new_df.columns = ['Entry', 'EC number', 'Sequence'] + list(range(0, max_ecs))

        if save:
            new_df.to_csv(os.path.join(self.output_folder, f'ProteInfer_{test_label}_protein_test_results_df.csv'), index=False)
        return new_df
