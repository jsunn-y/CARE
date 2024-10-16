import numpy as np
import pandas as pd
import random
import os

#supress warnings
import warnings
warnings.filterwarnings("ignore")

for baseline in  ['Similarity', 'CLIPZyme', 'CREEP', 'CREEP_text']:
    for split in ['easy', 'medium', 'hard']:

        if baseline == 'Similarity':
            reaction_similarities = np.load('Similarity/output/{}_split/retrieval_results/{}_reaction_test_reaction2reaction_retrieval_similarities.npy'.format(split, split))
        elif 'CREEP' in baseline:
            reaction_similarities = np.load('CREEP/output/{}_split/retrieval_results/{}_reaction_test_reaction2protein_retrieval_similarities.npy'.format(split, split))
            if 'text' in baseline:
                text_similarities = np.load('CREEP/output/{}_split/retrieval_results/{}_reaction_test_text2protein_retrieval_similarities.npy'.format(split, split))
        elif baseline == 'CLIPZyme':
            reaction_similarities = np.load('CLIPZyme/output/{}_split/retrieval_results/{}_reaction_test_reaction2protein_retrieval_similarities.npy'.format(split, split))
        

        query_df = pd.read_csv('../splits/task2/{}_reaction_test.csv'.format(split))
        query_EC_list = query_df['EC number'].values
        num_cols = len(query_df.columns)

        #add len(reaction_similarities) as empty columns to the query_df
        for i in range(reaction_similarities.shape[1]):
            query_df[i] = np.nan
        
        reference_EC_list = np.loadtxt('../processed_data/EC_list.txt', dtype=str)

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
            #append df['EC'] to the query_df in the numerical columns

            ### use this if similarity is based on euclidean distance rather than dot product ###
            #df.sort_values('overall rank', ascending=False, inplace=True)
            if baseline == 'Random':
                np.random.seed(42)
                query_df.iloc[i, num_cols:] = df['EC'].sample(frac=1).values
            else:
                query_df.iloc[i, num_cols:] = df['EC'].values
        
        #ensure the directory exists
        
        if not os.path.exists('results_summary/{}'.format(baseline)):
            os.makedirs('results_summary/{}'.format(baseline))

        query_df.to_csv('results_summary/{}/{}_reaction_test_results_df.csv'.format(baseline,split), index=False)

for baseline in  ['random']:
    for split in ['easy', 'medium', 'hard']:
        np.random.seed(42)
        num_ecs = 50
        rows = []
        df = pd.read_csv('../splits/task2/{}_reaction_test.csv'.format(split))
        train_ecs = set(pd.read_csv(os.path.join('../processed_data/reaction2EC.csv'))['EC number'].values)

        for entry, seq, true_ecs in df[['Reaction Text', 'Reaction', 'EC number']].values:
            # Randomly sample without replacement an entry from the training df
            predicted_ec = random.sample(train_ecs, k=num_ecs)
            rows.append([entry, true_ecs, seq] + predicted_ec)

        new_df = pd.DataFrame(rows)
        new_df.columns = ['Reaction Text', 'EC number', 'Reaction'] +  [str(s) for s in range(0, num_ecs)]

        os.makedirs('results_summary/{}'.format(baseline), exist_ok=True)

        new_df.to_csv('results_summary/{}/{}_reaction_test_results_df.csv'.format(baseline,split), index=False)