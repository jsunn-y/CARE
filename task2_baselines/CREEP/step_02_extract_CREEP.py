import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from torch.utils.data import DataLoader

from CREEP.models import SingleModalityModel
from CREEP.datasets import SingleModalityDataset
from CREEP.utils.tokenization import SmilesTokenizer

@torch.no_grad()
def extract(dataloader, AMP=True):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    repr_list, seq_list = [], []
    for batch_idx, batch in enumerate(L):
        seq = batch["sequence"]
        sequence_input_ids = batch["sequence_input_ids"].to(device)
        sequence_attention_mask = batch["sequence_attention_mask"].to(device)
        
        if AMP:
            with torch.cuda.amp.autocast():
                repr = model(sequence_input_ids, sequence_attention_mask)
        else:
            repr = model(sequence_input_ids, sequence_attention_mask)

        #TODO: make this work with text and reaction
        # if args.use_facilitator:
        #     description_repr = facilitator_distribution_model.inerence(description_repr)

        repr_list.append(repr.detach().cpu().numpy())

    repr_array = np.concatenate(repr_list)
    return repr_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    # parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--dataset", type=str, default="easy_reaction_test")
    parser.add_argument("--modality", type=str, default="protein", choices=["protein", "reaction", "text"])
    parser.add_argument("--protein_backbone_model", type=str, default="ProtT5", choices=["ProtT5"])
    parser.add_argument("--text_backbone_model", type=str, default="SciBERT")
    parser.add_argument("--reaction_backbone_model", type=str, default="rxnfp")
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    parser.add_argument("--reaction_max_sequence_len", type=int, default=512)
    parser.add_argument("--get_cluster_centers", dest="get_cluster_centers", action="store_true")
    parser.set_defaults(run_clustering=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=True)

    parser.add_argument("--pretrained_folder", type=str, default=None)

    args = parser.parse_args()
    print("arguments", args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.modality == "protein":
        args.backbone_model = args.protein_backbone_model
        args.max_sequence_len = args.protein_max_sequence_len
        ##### Load pretrained protein model
        protein_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, cache_dir="../../CREEP/data/pretrained_ProtT5")
        protein_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", cache_dir="../../CREEP/data/pretrained_ProtT5")
        protein_dim = 1024
        input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
        print("Loading protein model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        protein_model.load_state_dict(state_dict)

        ##### Load pretrained protein2latent model
        protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
        input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
        print("Loading protein2latent model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        protein2latent_model.load_state_dict(state_dict)

        tokenizer = protein_tokenizer
        modality_model = protein_model
        modality2latent_model = protein2latent_model

    elif args.modality == "reaction":
        args.backbone_model = args.reaction_backbone_model
        args.max_sequence_len = args.reaction_max_sequence_len
        #### Load pretrained reaction model
        reaction_tokenizer = SmilesTokenizer.from_pretrained("../../CREEP/data/pretrained_rxnfp/vocab.txt")
        reaction_model = BertModel.from_pretrained("../../CREEP/data/pretrained_rxnfp")
        reaction_dim = 256
        input_model_path = os.path.join(args.pretrained_folder, "reaction_model.pth")
        print("Loading reaction model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        reaction_model.load_state_dict(state_dict)

        ##### Load pretrained reaction2latent model
        reaction2latent_model = nn.Linear(reaction_dim, args.SSL_emb_dim)
        input_model_path = os.path.join(args.pretrained_folder, "reaction2latent_model.pth")
        print("Loading reaction2latent model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        reaction2latent_model.load_state_dict(state_dict)

        tokenizer = reaction_tokenizer
        modality_model = reaction_model
        modality2latent_model = reaction2latent_model

    elif args.modality == "text":
        args.backbone_model = args.text_backbone_model
        args.max_sequence_len = args.text_max_sequence_len
        ##### Load pretrained text model
        text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../CREEP/data/pretrained_SciBert")
        text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../CREEP/data/pretrained_SciBert")
        text_dim  = 768
        input_model_path = os.path.join(args.pretrained_folder, "text_model.pth")
        print("Loading text model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        text_model.load_state_dict(state_dict)

        ##### Load pretrained text2latent model
        text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
        input_model_path = os.path.join(args.pretrained_folder, "text2latent_model.pth")
        print("Loading text2latent model from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        text2latent_model.load_state_dict(state_dict)

        tokenizer = text_tokenizer
        modality_model = text_model
        modality2latent_model = text2latent_model

    # #### Load pretrained reaction2protein_facilitator model
    # reaction2protein_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    # state_dict = torch.load(os.path.join(args.pretrained_folder, "reaction2protein_facilitator_model.pth"), map_location='cpu')
    # reaction2protein_facilitator_model.load_state_dict(state_dict)

    # #### Load pretrained protein2reaction_facilitator model
    # protein2reaction_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    # state_dict = torch.load(os.path.join(args.pretrained_folder, "protein2reaction_facilitator_model.pth"), map_location='cpu')
    # protein2reaction_facilitator_model.load_state_dict(state_dict)

    model = SingleModalityModel(modality_model, modality2latent_model, args.backbone_model, args.modality)
    model.eval()
    model.to(device)

    if args.dataset == "all_proteins":
        file="../../processed_data/protein2EC_clustered50.csv" #used the ones clustered at 50% identity to speed things up
        df = pd.read_csv(file)

        #don't subsample to only unique indices because this will remove some of the EC classes
        # unique_protein_indices = np.loadtxt("../../processed_data/unique_protein_indices.txt", dtype=int)
        # df = df.iloc[unique_protein_indices].reset_index()
    else:
        file="../../splits/task2/{}.csv".format(args.dataset)
        df = pd.read_csv(file)

    ##### Load pretrained facilitator model
    # if args.use_facilitator:
    #     step_03_folder = os.path.join(args.pretrained_folder, "step_03_Gaussian_10")
    #     facilitator_distribution_model = GaussianFacilitatorModel(args.SSL_emb_dim)
    #     # TODO: will check later.
    #     input_model_path = os.path.join(step_03_folder, "facilitator_distribution_model.pth")
    #     print("Loading facilitator_distribution model from {}...".format(input_model_path))
    #     state_dict = torch.load(input_model_path, map_location='cpu')
    #     facilitator_distribution_model.load_state_dict(state_dict)
    #     facilitator_distribution_model = facilitator_distribution_model.to(device)
    #     facilitator_distribution_model.eval()

    dataset = SingleModalityDataset(
        file=file,
        tokenizer=tokenizer,
        max_sequence_len=args.max_sequence_len,
        modality=args.modality
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    repr_array = extract(dataloader)

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, "representations")
    os.makedirs(output_folder, exist_ok=True)

    if args.get_cluster_centers == False:
    
        saved_file_path = os.path.join(output_folder, args.dataset + "_representations")

        #if the file exists, load it
        if os.path.exists(saved_file_path + ".npy"):
            results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
        else:
            results = {}

        #TODO: don't save all the tensors if you don't need them all
        if args.modality == "protein":
            results["protein_repr_array"] = repr_array
        elif args.modality == "reaction":
            results["reaction_repr_array"] = repr_array
        elif args.modality == "text":
            results["text_repr_array"] = repr_array
            
        np.save(saved_file_path, results)

    # repr_array = np.load(saved_file_path + ".npy", allow_pickle=True).item()["protein_repr_array"]
    else:
        df['index'] = df.index
        ec2index = df.groupby('EC number')['index'].apply(list).to_frame().to_dict()['index']
        EClist = np.loadtxt("../../processed_data/EC_list.txt", dtype=str)

        print(len(EClist))
        print(len(ec2index.keys()))

        #temporary line to just check if the code runs
        #EClist = [ec for ec in EClist if ec in ec2index.keys()]
        #assert len(EClist) == len(ec2index.keys())
        
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

        if args.modality == "protein":
            results["protein_repr_array"] = cluster_centers
        elif args.modality == "reaction":
            results["reaction_repr_array"] = cluster_centers
        elif args.modality == "text":
            results["text_repr_array"] = cluster_centers

        print(cluster_centers.shape)
        np.save(saved_file_path, results)