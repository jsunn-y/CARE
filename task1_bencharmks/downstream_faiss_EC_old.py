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

from utils import TextDataset, TextProteinPairDataset, evaluate
from ProteinDT.models import ProteinTextModel, GaussianFacilitatorModel
from ProteinDT.datasets import SwissProtCLAPDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--reference_dataset", type=str, default='SwissProtEnzymeCLAP')
    parser.add_argument("--evaluate_dataset", type=str, default='SwissProtEnzymeCLAP')
    parser.add_argument("--protein_backbone_model", type=str, default="ProtBERT_BFD", choices=["ProtBERT", "ProtBERT_BFD"])
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)
        
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])
    #parser.add_argument("--use_full_dataset", dest="use_full_dataset", action="store_true")
    # parser.set_defaults(use_full_dataset=True)

    args = parser.parse_args()
    print("arguments", args)
    step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_folder = 'EC_classification'
    if 'reaction' in args.pretrained_folder:
        reaction = True
    else:
        reaction = False

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, data_folder)
    os.makedirs(output_folder, exist_ok=True)

    #all_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/enzyme_inner.csv', index_col=0)
    # if reaction:
    #     indices = all_df['index2'].values
    # else:
    #     indices = all_df['index1'].values 
    
    root = args.pretrained_folder + "/step_02_extract_representation/"

    if args.evaluate_dataset == 'SwissProtEnzymeCLAP':
        evaluate_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/EnzymeCLAP_240319.csv', index_col=0)
    else:
        evaluate_df = pd.read_csv('../../data/test_sets/' + args.evaluate_dataset + '.csv')

    if args.reference_dataset == 'SwissProtEnzymeCLAP':
        reference_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/EnzymeCLAP_240319.csv', index_col=0)
    elif 'trembl' in args.reference_dataset:
        reference_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/{}.csv'.format(args.reference_dataset), index_col=0)
    else:
        if 'cluster_centers' in args.reference_dataset:
            reference_df = evaluate_df
        else:
            reference_df = pd.read_csv('../../data/test_sets/' + args.reference_dataset + '.csv')
    #unique_indices = all_df.drop_duplicates(subset='reaction_smiles').index # easier to look for this 
    indices = np.arange(len(reference_df))

    #index1 to subset from swissprot representation, index2 to subset from ariane representation

    #load the reference representations and subset to enzymes
    
    # if reaction:
    reference_representation_file = os.path.join(root, args.reference_dataset + "_representations.npz")
    # else:
    #     representation_file = root + "/step_02_pairwise_representation/pairwise_representation.npz"
    reference_representation_data = np.load(reference_representation_file)

    #load the reference representations
    reference_protein_repr_array = reference_representation_data["protein_repr_array"][indices]


    #load the representations to be evaluated
    evaluate_representation_file = os.path.join(root, args.evaluate_dataset + "_representations.npz")
    evaluate_representation_data = np.load(evaluate_representation_file)

    evaluate_protein_repr_array = evaluate_representation_data["protein_repr_array"]
    

    d = reference_protein_repr_array.shape[1]  # dimension

    #res = faiss.StandardGpuResources()

    index_protein = faiss.IndexFlatIP(d)   # build the index using inner product similarity
    #index_protein = faiss.index_cpu_to_gpu(res, 0, index_protein)
    #index_protein  = faiss.IndexFlatL2(d)   # alternatively build the index using L2 similarity
    index_protein.add(reference_protein_repr_array)

    k = 1 #give the k nearest neighbors
    n_test=len(evaluate_protein_repr_array)

  
    #Method 2 (protein-protein similarity)
    D, I = index_protein.search(evaluate_protein_repr_array[:n_test], k) #queries are the proteins, finds similarity to protein embeddings, returns nearest neighbors in reference dataset

    #print(D.shape)
    #print(I.shape)
    print(I)
    print(D)
    #take the corresponding indices from the reference text txt
    
    reference_protein_EC_list = reference_df['brenda'].values
    evaluate_protein_EC_list = evaluate_df['brenda'].values

    corrects = []
    predicted_ECs = []
    for i, evaluate_EC in enumerate(evaluate_protein_EC_list[:n_test]):
        nearest_ECs = reference_protein_EC_list[I[i]]
        #print(nearest_desc_sequences)
        correct = evaluate_EC in nearest_ECs
        corrects.append(correct)
        predicted_ECs.append(nearest_ECs[0])

    accuracy = np.mean(corrects)
    #unique_accuracy = np.mean(np.array(corrects[unique_indices]))
    if len(reference_protein_EC_list) == len(predicted_ECs):
        results = pd.DataFrame({'truth': reference_protein_EC_list, 'predicted': predicted_ECs, 'correct': corrects})
        results.to_csv(os.path.join(output_folder, args.evaluate_dataset + "_EC_results.csv"))
    print(accuracy)
    
    

