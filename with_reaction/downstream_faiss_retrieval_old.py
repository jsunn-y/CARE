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
    parser.add_argument("-k", type=int, default=10) #number to evaluate
    
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)
        
    parser.add_argument("--pretrained_folder", type=str, default=None)
    parser.add_argument("--facilitator_distribution", type=str, default="Gaussian", choices=["Gaussian"])
    parser.add_argument("--input", type=str, default="reaction", choices=["text", "reaction", "protein"])
    parser.add_argument("--output", type=str, default="protein", choices=["text", "reaction", "protein"])
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

    data_folder = 'similarity_search'
    # if args.descriptor == 'reaction':
    #     reaction = True
    # else:
    #     reaction = False

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, data_folder)
    os.makedirs(output_folder, exist_ok=True)

    #all_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/enzyme_inner.csv', index_col=0)
    # if reaction:
    #     indices = all_df['index2'].values
    # else:
    #     indices = all_df['index1'].values 

    if args.reference_dataset == 'SwissProtEnzymeCLAP':
        reference_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/EnzymeCLAP_240319.csv', index_col=0)
    else:
        reference_df = pd.read_csv('../../data/test_sets/' + args.reference_dataset + '.csv')
    #unique_indices = all_df.drop_duplicates(subset='reaction_smiles').index # easier to look for this 
    indices = np.arange(len(reference_df))
    
    root = args.pretrained_folder + "/step_02_extract_representation/"

    if args.evaluate_dataset == 'SwissProtEnzymeCLAP':
        evaluate_df = pd.read_csv('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/EnzymeCLAP_240319.csv', index_col=0)
    else:
        evaluate_df = pd.read_csv('../../data/test_sets/' + args.evaluate_dataset + '.csv')

    #index1 to subset from swissprot representation, index2 to subset from ariane representation

    #load the reference representations and subset to enzymes
    
    # if reaction:
    reference_representation_file = os.path.join(root, args.reference_dataset + "_representations.npz")
    # else:
    #     representation_file = root + "/step_02_pairwise_representation/pairwise_representation.npz"
    reference_representation_data = np.load(reference_representation_file)

    #load the reference representations
    reference_protein_repr_array = reference_representation_data["protein_repr_array"][indices]
    if "text_repr_array" in reference_representation_data:
        reference_text_repr_array = reference_representation_data["text_repr_array"][indices]
    else:
        reference_text_repr_array = reference_representation_data["description_repr_array"][indices]

    if reaction:
        reference_reaction_repr_array = reference_representation_data["reaction_repr_array"][indices]

    #load the representations to be evaluated
    evaluate_representation_file = os.path.join(root, args.evaluate_dataset + "_representations.npz")
    evaluate_representation_data = np.load(evaluate_representation_file)

    evaluate_protein_repr_array = evaluate_representation_data["protein_repr_array"]
    if "text_repr_array" in evaluate_representation_data:
        evaluate_text_repr_array = evaluate_representation_data["text_repr_array"]
    else:
        evaluate_text_repr_array = evaluate_representation_data["description_repr_array"]
    if reaction:
        if "GCLAP" in args.pretrained_folder:
            evaluate_reaction_repr_array = evaluate_representation_data["reaction2protein_repr_array"]
        else:
            evaluate_reaction_repr_array = evaluate_representation_data["reaction_repr_array"]

    d = reference_protein_repr_array.shape[1]  # dimension

    #res = faiss.StandardGpuResources()

    index_protein = faiss.IndexFlatIP(d)   # build the index using inner product similarity
    #index_protein = faiss.index_cpu_to_gpu(res, 0, index_protein)
    #index_protein  = faiss.IndexFlatL2(d)   # alternatively build the index using L2 similarity
    index_protein.add(reference_protein_repr_array)

    if reaction:
        index_reaction = faiss.IndexFlatIP(d)
        #index_reaction = faiss.index_cpu_to_gpu(res, 0, index_reaction)
        #index_text  = faiss.IndexFlatL2(d) 
        index_reaction.add(reference_reaction_repr_array)
        index_desc = index_reaction
        evaluate_desc_repr_array = evaluate_reaction_repr_array
    else:
        index_text = faiss.IndexFlatIP(d)
        #index_text  = faiss.IndexFlatL2(d) 
        #index_text = faiss.index_cpu_to_gpu(res, 0, index_text)
        index_text.add(reference_text_repr_array)
        index_desc = index_text
        evaluate_desc_repr_array = evaluate_text_repr_array

    k = args.k #give the 20 nearest neighbors
    n_test=len(evaluate_protein_repr_array)

    #Method 1 (text-text similarity) - this should have 100% retrieval
    #D, I = index_desc.search(evaluate_desc_repr_array[:n_test], k) #queries are the texts, finds similarity to text embeddings, returns nearest neighbors in reference dataset
  
    #Method 2 (protein-text/reaction similarity)
    D, I = index_protein.search(evaluate_desc_repr_array[:n_test], k) #queries are the texts, finds similarity to protein embeddings, returns nearest neighbors in reference dataset

    #print(D.shape)
    #print(I.shape)
    print(I)
    print(D)
    #take the corresponding indices from the reference text txt
    reference_text_seq_list = reference_df['reaction_eq'].values
    reference_desc_seq_list = reference_text_seq_list
    reference_protein_seq_list = reference_df['sequence'].values
    if reaction:
        reference_reaction_seq_list = reference_df['reaction_smiles'].values
        reference_desc_seq_list = reference_reaction_seq_list
    
    evaluate_text_seq_list = evaluate_df['reaction_eq'].values
    evaluate_protein_seq_list = evaluate_df['sequence'].values

    if reaction:
        evaluate_reaction_seq_list = evaluate_df['reaction_smiles'].values
        evaluate_desc_seq_list = evaluate_reaction_seq_list
    else:
        evaluate_desc_seq_list = evaluate_text_seq_list

    sequence_identities = []
    corrects = []
    for i, (desc_match, protein_match) in enumerate(zip(evaluate_desc_seq_list[:n_test], evaluate_protein_seq_list[:n_test])):
        #print(text_match)
        #print(I[i])
        nearest_desc_sequences = reference_desc_seq_list[I[i]]
        #print(nearest_desc_sequences)
        correct = desc_match in nearest_desc_sequences
        corrects.append(correct)

        nearest_text_sequences = reference_text_seq_list[I[i]]
        #print(nearest_text_sequences)
        #print(text_match in nearest_text_sequences)
        protein_top_hit = reference_protein_seq_list[I[i][0]]
        seq_identity = 1-levenshtein_distance(protein_match, protein_top_hit)/max(len(protein_match), len(protein_top_hit))
        sequence_identities.append(seq_identity)
    accuracy = np.mean(corrects)
    #unique_accuracy = np.mean(np.array(corrects[unique_indices]))

    pd.DataFrame({'correct': corrects, 'sequence_identity': sequence_identities}).to_csv(os.path.join(output_folder, args.evaluate_dataset + "_similarity_results.csv"))
    print(accuracy)
    #print(unique_accuracy)
    print(sequence_identities[-10:])
    
    

