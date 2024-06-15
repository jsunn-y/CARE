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
from drfp import DrfpEncoder


u = SciUtil()
seed=42

        
    
# ----------------------------------------------------------------------------------
#                   Class that calls each tool.
# ----------------------------------------------------------------------------------
class Task2:

    def __init__(self, data_folder, output_folder, processed_dir, pretrained_dir=None):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.processed_dir = processed_dir # Where the default files processed by CARE are
        ec_to_text = pd.read_csv(f'{self.processed_dir}text2EC.csv')
        self.ec_to_text = dict(zip(ec_to_text['EC number'], ec_to_text['Text']))
        self.pretrained_dir = pretrained_dir
        
    def tabulate_results(self, baseline, split):
        if baseline == 'Similarity':
            reaction_similarities = np.load('{}Similarity/output/{}_split/retrieval_results/{}_reaction_test_reaction2reaction_retrieval_similarities.npy'.format(self.pretrained_dir, split, split))
        elif 'CREEP' in baseline:
            reaction_similarities = np.load('{}CREEP/output/{}_split/retrieval_results/{}_reaction_test_reaction2protein_retrieval_similarities.npy'.format(self.pretrained_dir, split, split))
            if 'text' in baseline:
                text_similarities = np.load('{}CREEP/output/{}_split/retrieval_results/{}_reaction_test_text2protein_retrieval_similarities.npy'.format(self.pretrained_dir, split, split))
        elif baseline == 'CLIPZyme':
            reaction_similarities = np.load('{}CLIPZyme/output/{}_split/retrieval_results/{}_reaction_test_reaction2protein_retrieval_similarities.npy'.format(self.pretrained_dir, split, split))
        
        query_df = self.get_test_df(split)
        query_EC_list = query_df['EC number'].values
        num_cols = len(query_df.columns)

        #add len(reaction_similarities) as empty columns to the query_df
        for i in range(reaction_similarities.shape[1]):
            query_df[i] = np.nan
        
        reference_EC_list = self.get_ec_list()

        #get the number of columns in the query_df
        for i, query_EC in enumerate(query_EC_list):
            reaction_similarity = reaction_similarities[i]
            
            df = pd.DataFrame({'EC': reference_EC_list, 'reaction similarity': reaction_similarity})
            df['reaction rank'] = df['reaction similarity'].rank(ascending=False)
            if 'text' in baseline:
                text_similarity = text_similarities[i]
                df['text similarity'] = text_similarity
                df['text rank'] = df['text similarity'].rank(ascending=False)
                df['overall rank'] = (df['reaction rank'] + df['text rank'])/2
            else:
                df['overall rank'] = df['reaction rank']
                
            df.sort_values('overall rank', ascending=True, inplace=True) #need to check if this order is correct
            ### use this if similarity is based on euclidean distance rather than dot product ###
            if baseline == 'Random':
                np.random.seed(42)
                query_df.iloc[i, num_cols:] = df['EC'].sample(frac=1).values
            else:
                query_df.iloc[i, num_cols:] = df['EC'].values
        
        #ensure the directory exists
        
        if not os.path.exists('{}/results_summary/{}'.format(self.output_folder, baseline)):
            os.makedirs(f'{self.output_folder}results_summary/{baseline}')

        query_df.to_csv('{}results_summary/{}/{}_reaction_test_results_df.csv'.format(self.output_folder, baseline,split), index=False)
        return query_df

    def get_ec2text(self):
        return self.ec_to_text
    
    def get_ec_list(self):
        return np.loadtxt(f'{self.processed_dir}EC_list.txt', dtype=str)

    def get_proteins(self):
        return pd.read_csv(f'{self.processed_dir}protein2EC_clustered50.csv')
    
    def get_reactions(self):
        return pd.read_csv(f'{self.processed_dir}reaction2EC.csv')
    
    def get_ranking_for_reaction(self, reaction_smiles: str, name: str):
        """
        For a specific reaction get the similarities from DRFP and CREEP.

        Name is the name of your reaction.
        """
        template = self.get_test_df("easy")
        df = template.iloc[0:1]
        # ToDo: potentially check the viability of the smiles
        df['Reaction'] = reaction_smiles
        #drop Mapped Reaction
        df = df.drop(columns=['Mapped Reaction', 'Reaction Text', 'Duplicated EC', 'Reactions with a single EC'])
        df['EC number'] = '0.0.0.0'
        df['EC3'] = '0.0.0'
        df['EC2'] = '0.0'
        df['EC1'] = '0'
        query_reactions = df['Reaction'].values

        fps = DrfpEncoder.encode(query_reactions, show_progress_bar=True)

        fps = np.vstack(fps)
        output_folder = '{}Similarity/output/{}_split/representations/'.format(self.pretrained_dir, name)
        os.makedirs(output_folder, exist_ok=True)
        split = name
        saved_file_path = os.path.join('{}/{}_reaction_test_representations'.format(output_folder, split))
        #if the file exists, load it
        if os.path.exists(saved_file_path + ".npy"):
            results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
        else:
            results = {}

        results["reaction_repr_array"] = fps  
        np.save(saved_file_path, results)
        df = self.get_reactions()

        reactions = df['Reaction'].values

        fps = DrfpEncoder.encode(reactions, show_progress_bar=True)
        repr_array = np.vstack(fps)

        df['index'] = df.index
        ec2index = df.groupby('EC number')['index'].apply(list).to_frame().to_dict()['index']
        EClist = self.get_ec_list()

        cluster_centers = np.zeros((len(EClist), repr_array.shape[1]))
        for i, ec in enumerate(EClist):
            #average together the embeddings for each EC number
            if ec in ec2index.keys():
                indices = ec2index[ec]
                cluster_centers[i] = np.mean(repr_array[indices], axis=0)

        saved_file_path = os.path.join(output_folder, "all_ECs_cluster_centers")
        #if the file exists, load it
        if os.path.exists(saved_file_path + ".npy"):
            results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
        else:
            results = {}

        results["reaction_repr_array"] = cluster_centers

        print(cluster_centers.shape)
        np.save(saved_file_path, results)

        # Now we can do the downstream retrieval!     
        return self.downstream_retrieval('Similarity', 'all_ECs', name, 'reaction', 'reaction')
    
    def downstream_retrieval(self, baseline, reference_dataset, query_dataset, reference_modality, query_modality, k=10, seed=42):
        """ Once a model has been pretrained we can retrieve the data using a systematic and consistent format. """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        if reference_dataset == 'all_ECs':
            reference_EC_list = self.get_ec_list()
        if reference_dataset == 'all_proteins':
            reference_df = self.get_proteins()
            reference_EC_list = reference_df['EC number'].values
        elif reference_dataset == 'all_reactions':
            reference_df = self.get_reactions()
            reference_EC_list = reference_df['EC number'].values

        root = self.pretrained_dir + f"{baseline}/output/{query_dataset}_split/representations/"

        query_df = self.get_test_df(query_dataset)

        #load the reference representations
        if reference_dataset=='all_ECs':
            reference_representation_file = os.path.join(root, reference_dataset + "_cluster_centers.npy")
        else:
            reference_representation_file = os.path.join(root, reference_dataset + "_representations.npy")
        reference_representation_data = np.load(reference_representation_file, allow_pickle=True).item()
        
        #load the representations to be queried
        query_representation_file = os.path.join(root, query_dataset + "_" + query_modality + "_test_representations.npy")
        query_representation_data = np.load(query_representation_file, allow_pickle=True).item()

        reference_key = reference_modality + '_repr_array'
        query_key = query_modality + '_repr_array'

        reference_repr_array = reference_representation_data[reference_key]
        query_repr_array = query_representation_data[query_key]

        d = reference_repr_array.shape[1]  #dimension

        ec2text = self.get_ec2text()

        if query_modality == 'text':
            query_df['Text'] = query_df['EC number'].map(ec2text)

        modality2column_dict = {'protein': 'Sequence', 'text': 'Text', 'reaction': 'Reaction'}

        query_inmodality_list = query_df[modality2column_dict[query_modality]].values #not sure if this is still used
        query_EC_list = query_df['EC number'].values
        
        all_indices = np.zeros((len(query_EC_list), len(reference_EC_list)))
        all_similarities = np.zeros((len(query_EC_list), len(reference_EC_list)))

        for i, (query_inmodality, query_EC) in enumerate(zip(query_inmodality_list, query_EC_list)):
            #compute the dot product similarity between the query and the reference
            query_repr = query_repr_array[i].reshape(1, -1)
            normalization = np.linalg.norm(query_repr) * np.linalg.norm(reference_repr_array, axis=1)
            #replace zeros with 1
            normalization[normalization == 0] = 1 #normalizing to cosine similarity doesn't work when there are too many zeros
            similarity = np.dot(query_repr, reference_repr_array.T)/normalization 

            #check if the query is in the reference
            query_EC_index = np.where(reference_EC_list == query_EC)[0]
            if len(query_EC_index) == 0:
                print("Query EC is not in the reference ECs.")
            else:
                query_EC_index = query_EC_index[0]
            
            #print(query_EC_index)
            sorted_indices = np.argsort(similarity, axis=1)[:, ::-1][0] #sort in descending order
            all_indices[i] = sorted_indices
            all_similarities[i] = similarity

        np.save(os.path.join(self.output_folder, query_dataset + "_" + query_modality + "2" + reference_modality + "_retrieval_similarities.npy"), all_similarities)

        return all_similarities
    
    def get_train_df(self, label):
        return pd.read_csv(os.path.join(self.data_folder, f'{label}_reaction_train.csv'))
    
    def get_all_ecs(self):
        return set(pd.read_csv(os.path.join(self.data_folder, f'reaction2EC.csv'))['EC number'].values)

    def get_test_df(self, label):
        return pd.read_csv(os.path.join(self.data_folder, f'{label}_reaction_test.csv'))
        
    def get_similarity(self, test_label, encode=False, df=None):
        """
        Encode the reactions using Drfp
        """
        query_df = pd.read_csv(self.get_test_df(test_label)) if df is None else df
        query_reactions = query_df['Reaction'].values
        if encode:
            fps = DrfpEncoder.encode(query_reactions, show_progress_bar=True)

            fps = np.vstack(fps)
            os.makedirs(f'{self.output_folder}representations/', exist_ok=True)

            saved_file_path = os.path.join(f'{self.output_folder}representations/', f'{test_label}_reaction_test_representations')

            #if the file exists, load it
            if os.path.exists(saved_file_path + ".npy"):
                results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
            else:
                results = {}

            results["reaction_repr_array"] = fps
                
            np.save(saved_file_path, results)
        

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