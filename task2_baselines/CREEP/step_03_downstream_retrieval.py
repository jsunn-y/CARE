import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time
import faiss
import pandas as pd
from Levenshtein import distance as levenshtein_distance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

#from utils import TextDataset, TextProteinPairDataset, evaluate
# from CREEP.models import ProteinTextModel, GaussianFacilitatorModel
# from CREEP.datasets import SwissProtCLAPDataset

# def print_results(args, label, indices):
#     print('\n' + label + ' N=' + str(len(indices)))
#     print('{} to {} retreival accuracy (k={}): {}'.format(args.query_modality, args.reference_modality, args.k, np.mean(corrects[indices])))
#     print('Average EC ranking to find: {}'.format(np.mean(rankings[indices]))) 
#     print('EC Classification accuracy (k=1): {}'.format(np.mean(np.array(query_EC_list)[indices] == np.array(predicted_ECs)[indices])))
#     #print('Average sequence identity: {}'.format(np.mean(sequence_identities[indices])))
#     return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_dataset", type=str, choices=['easy_reaction', 'medium_reaction', 'hard_reaction'])
    parser.add_argument("--reference_dataset", type=str, default='all_ECs', choices=['all_ECs', 'all_proteins', 'all_reactions'])

    parser.add_argument("-k", type=int, default=10) #number to evaluate

    parser.add_argument("--use_cluster_center", dest="use_cluster_center", action="store_true")
    parser.set_defaults(use_cluster_center=True)
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--query_modality", type=str, default="reaction", choices=["text", "reaction", "protein"])
    parser.add_argument("--reference_modality", type=str, default="protein", choices=["text", "reaction", "protein"])

    args = parser.parse_args()
    print("arguments", args)
    # step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_folder = 'retrieval_results'

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, data_folder)
    os.makedirs(output_folder, exist_ok=True)

    if args.reference_dataset == 'all_ECs':
        reference_EC_list = np.loadtxt('../../processed_data/EC_list.txt', dtype=str)
    if args.reference_dataset == 'all_proteins':
        reference_df = pd.read_csv('../../processed_data/protein2EC.csv')
        unique_indices = np.loadtxt('../../processed_data/unique_protein_indices.txt', dtype=int)
        reference_df = reference_df.iloc[unique_indices]
        reference_EC_list = reference_df['EC number'].values
    elif args.reference_dataset == 'all_reactions':
        reference_df = pd.read_csv('../../processed_data/reaction2EC.csv')
        reference_EC_list = reference_df['EC number'].values
    
    root = args.pretrained_folder + "/representations/"
    query_df = pd.read_csv('../../splits/task2/' + args.query_dataset + '_test.csv')

    #load the reference representations
    if args.use_cluster_center:
        reference_representation_file = os.path.join(root, args.reference_dataset + "_cluster_centers.npy")
    else:
        reference_representation_file = os.path.join(root, args.reference_dataset + "_representations.npy")
    reference_representation_data = np.load(reference_representation_file, allow_pickle=True).item()
    
    #load the representations to be queried
    query_representation_file = os.path.join(root, args.query_dataset + "_representations.npy")
    query_representation_data = np.load(query_representation_file, allow_pickle=True).item()

    reference_key = args.reference_modality + '_repr_array'
    query_key = args.query_modality + '_repr_array'

    reference_repr_array = reference_representation_data[reference_key]
    query_repr_array = query_representation_data[query_key]

    d = reference_repr_array.shape[1]  #dimension

    ec2text = pd.read_csv('../../processed_data/text2EC.csv').set_index('EC number').to_dict()['Text']

    if args.query_modality == 'text':
        query_df['Text'] = query_df['EC number'].map(ec2text)

    modality2column_dict = {'protein': 'Sequence', 'text': 'Text', 'reaction': 'Reaction'}

    k = args.k #find the k nearest neighbors

    query_inmodality_list = query_df[modality2column_dict[args.query_modality]].values #not sure if this is still used
    query_EC_list = query_df['EC number'].values
    
    # sequence_identities = []
    # corrects = []
    # predicted_ECs = []
    # rankings = [] #keeps track of rnaking where the query is in
    all_indices = np.zeros((len(query_EC_list), len(reference_EC_list)))
    all_similarities = np.zeros((len(query_EC_list), len(reference_EC_list)))

    for i, (query_inmodality, query_EC) in enumerate(zip(query_inmodality_list, query_EC_list)):
        #compute the dot product similarity between the query and the reference
        query_repr = query_repr_array[i].reshape(1, -1)
        similarity = np.dot(query_repr, reference_repr_array.T)

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

        #print(sorted_indices)
        #print(np.where(sorted_indices == query_EC_index)[0][0])
        
        #rankings.append(np.where(sorted_indices == query_EC_index)[0][0] + 1)
        
        #print(positions)
        
        #top_indices = sorted_indices[:k] #take the top k indices

        # #nearest_queries_inmodality = reference_inmodality_list[I[i]]
        # nearest_ECs = reference_EC_list[top_indices]
        # predicted_ECs.append(reference_EC_list[top_indices[0]]) #for the top hit using EC

        # #correct = query_inmodality in nearest_queries_inmodality
        # correct = query_EC in nearest_ECs #for now measure correct 
        # corrects.append(correct)

        #protein_top_hit = reference_protein_list[top_indices[0]]

        #seq_identity = 1-levenshtein_distance(query_protein, protein_top_hit)/max(len(query_protein), len(protein_top_hit))
        #sequence_identities.append(seq_identity)

    # sequence_identities = np.array(sequence_identities)
    # corrects = np.array(corrects)
    # accuracy = np.mean(corrects)

    # rankings = np.array(rankings)

    #results = pd.DataFrame({'correct': corrects, 'sequence_identity': sequence_identities, 'true_EC': query_EC_list, 'predicted_EC': predicted_ECs, 'rankings': rankings})
    #results = pd.DataFrame({'correct': corrects, 'true_EC': query_EC_list, 'predicted_EC': predicted_ECs, 'rankings': rankings})
    
    # results = pd.DataFrame({'predicted_EC': predicted_ECs, 'rankings': rankings})
    # results.to_csv(os.path.join(output_folder, args.query_dataset + "_" + args.query_modality + "2" + args.reference_modality + "_retrieval_results.csv"))
    #np.save(os.path.join(output_folder, args.query_dataset + "_" + args.query_modality + "2" + args.reference_modality + "_retrieval_indices.npy"), all_indices)
    np.save(os.path.join(output_folder, args.query_dataset + "_" + args.query_modality + "2" + args.reference_modality + "_retrieval_similarities.npy"), all_similarities)
    
    #label = 'Average Performance'
    # print(len(corrects), len(sequence_identities), len(query_EC_list), len(predicted_ECs))
    #indices = np.arange(len(corrects)) #added this to just loop over everything
    #print(len(corrects))
    #print_results(args, label, indices)



        
    

