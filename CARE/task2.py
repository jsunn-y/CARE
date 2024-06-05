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

from openai import OpenAI
import numpy as np
import pandas as pd
from tqdm import tqdm
from sciutil import SciUtil
import os
import random

u = SciUtil()
seed=42

# ----------------------------------------------------------------------------------
#                   Class that calls each tool.
# ----------------------------------------------------------------------------------
class Task2:

    def __init__(self, data_folder, output_folder, ec2text):
        self.data_folder = data_folder
        self.output_folder = output_folder
        ec_to_text = pd.read_csv(ec2text)
        self.ec_to_text = dict(zip(ec_to_text['EC number'], ec_to_text['Text']))

    def get_train_df(self, label):
        return pd.read_csv(os.path.join(self.data_folder, f'{label}_reaction_train.csv'))
    
    def get_all_ecs(self):
        return set(pd.read_csv(os.path.join(self.data_folder, f'reaction2EC.csv'))['EC number'].values)

    def get_test_df(self, label):
        return pd.read_csv(os.path.join(self.data_folder, f'{label}_reaction_test.csv'))

    def get_ChatGPT(self, test_label, n=10, query_type='reaction', save=False, api_key=None, subsample=None, run_tag=''):
        """
        Gets the results for a series of ECs and formats it correctly for the paper
        """
        client = OpenAI(api_key=api_key)
        df = self.get_test_df(test_label)
        if subsample is not None: # Just for testing so we don't run too many
            df = df[df['Entry'].isin(subsample)]
        rows = []
        for entry, true_ec, text_annot, reaction in tqdm(df[['Reaction Text', 'EC number', 'Text', 'Reaction']].values):
            if query_type == 'reaction':
                text = f"Return the top {n} most likely EC numbers as a comma separated list for this reaction: {entry}."
                completion = client.chat.completions.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system",
                        "content": 
                        "You are protein engineer capable of predicting EC numbers from a reaction that corresponds to a specific enzyme."
                        + "You are also a skilled programmer and able to execute the code necessary to predict an EC number when you can't use reason alone." 
                        + "Given a reaction you are able to determine the most likely enzyme class for a reaction." 
                        + "You don't give up when faced with a reaction you don't know, you will use tools to resolve the most likely enzyme number."
                        + "You only return enzyme commission numbers in a comma separated list, no other text is returned, you have failed if you do "
                        + " not return the EC numbers. You only return the exact number of EC numbers that a user has provided requested, ordered by their likelihood of being correct."},
                        {"role": "user", "content": text}
                    ]
                )
            elif query_type == 'reaction+text':
                text = f"Return the top {n} most likely EC numbers as a comma separated list for this reaction: {entry}, which associates with the following text: {text_annot}."
                completion = client.chat.completions.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system",
                        "content": 
                        "You are protein engineer capable of predicting EC numbers from a combination of textual information and a reaction that corresponds to a specific protein."
                        + "You are also a skilled programmer and able to execute the code necessary to predict an EC number when you can't use reason alone." 
                        + "Given a reaction and text information of an EC you are able to determine the most likely enzyme class for a reaction." 
                        + "You don't give up when faced with a reaction you don't know, you will use tools to resolve the most likely enzyme number."
                        + "You only return enzyme commission numbers in a comma separated list, no other text is returned, you have failed if you do "
                        + " not return the EC numbers. You only return the exact number of EC numbers that a user has provided requested, ordered by their likelihood of being correct."},
                        {"role": "user", "content": text}
                    ]
                )
            print(completion.choices[0].message.content)
            preds = completion.choices[0].message.content.replace(" ", "").split(',')
            for p in preds:
                rows.append([entry, true_ec, p, seq]) # Costs you ~1c per query
        results = pd.DataFrame(rows)
        results.columns = ['entry',  'true_ecs', 'predicted_ecs', 'seq']
        grped = results.groupby('entry')
        max_ecs = 0
        rows = []
        # Get the raw test set and then build the new dataset based on that! 
        test_df = self.get_test_df(test_label)
        # Now we want to iterate through and get the predicted EC numbers
        entry_to_seq = dict(zip(test_df['Entry'], test_df['Sequence']))

        for query in test_df['Entry'].values:
            try:
                grp = grped.get_group(query)
                # Always will be the same for the grouped 
                true_ec = ';'.join(set([c for c in grp['true_ecs'].values]))
                seq = entry_to_seq.get(query) # Only returns one sequence when there could be multiple
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
            except:
                u.warn_p([query, ' Had no match from ChatGPT.'])
        new_df = pd.DataFrame(rows)
        new_df.columns = ['Entry', 'EC number', 'Text'] +  [str(s) for s in range(0, max_ecs)]

        # Save to a file in the default location
        if save:
            new_df.to_csv(os.path.join(self.output_folder, f'{run_tag}{test_label}_reaction_test_results_df.csv'), index=False)
            u.dp(["Done: ", test_label, "\nSaved to:", os.path.join(self.output_folder, f'{run_tag}{test_label}_reaction_test_results_df.csv')])

        return new_df
    
    def randomly_assign_EC(self, test_label, save=False, num_ecs=10, run_tag=''):
        """ Randomly assign n EC numbers from the training dataset """
        # For each test set we'll randomly set the EC classification.
        np.random.seed(42)
        rows = []
        df = self.get_test_df(test_label)
        train_ecs = self.get_all_ecs()

        for entry, seq, true_ecs in df[['Reaction Text', 'Reaction', 'EC number']].values:
            # Randomly sample without replacement an entry from the training df
            predicted_ec = random.sample(train_ecs, k=num_ecs)
            rows.append([entry, true_ecs, seq] + predicted_ec)

        new_df = pd.DataFrame(rows)
        new_df.columns = ['Reaction Text', 'EC number', 'Reaction'] +  [str(s) for s in range(0, num_ecs)]
        
        if save:
            new_df.to_csv(os.path.join(self.output_folder, f'{run_tag}{test_label}_reaction_test_results_df.csv'), index=False)
            u.dp(["Done: ", test_label, "\nSaved to:", os.path.join(self.output_folder, f'{run_tag}{test_label}_reaction_test_results_df.csv')])

        return new_df