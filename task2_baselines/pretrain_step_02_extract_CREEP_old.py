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

from ProteinDT.models import CREEPModel, AEFacilitatorModel
from ProteinDT.datasets import CREEPDataset
from ProteinDT.utils.tokenization import SmilesTokenizer

@torch.no_grad()
def extract(dataloader, AMP=True):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    protein_repr_list, text_repr_list, reaction_repr_list, reaction2protein_repr_list, protein2reaction_repr_list, protein_seq_list, text_seq_list, reaction_seq_list = [], [], [], [], [], [], [], []
    for batch_idx, batch in enumerate(L):
        protein_seq = batch["protein_sequence"]
        text_seq = batch["text_sequence"]
        reaction_seq = batch["reaction_sequence"]
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        reaction_sequence_input_ids = batch["reaction_sequence_input_ids"].to(device)
        reaction_sequence_attention_mask = batch["reaction_sequence_attention_mask"].to(device)
        
        if AMP:
            with torch.cuda.amp.autocast():
                protein_repr, text_repr, reaction_repr, reaction2protein_repr, protein2reaction_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask)
        else:
            protein_repr, text_repr, reaction_repr, reaction2protein_repr, protein2reaction_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask)

        #TODO: make this work with text and reaction
        # if args.use_facilitator:
        #     description_repr = facilitator_distribution_model.inerence(description_repr)

        protein_repr_list.append(protein_repr.detach().cpu().numpy())
        text_repr_list.append(text_repr.detach().cpu().numpy())
        reaction_repr_list.append(reaction_repr.detach().cpu().numpy())
        reaction2protein_repr_list.append(reaction2protein_repr.detach().cpu().numpy())
        protein2reaction_repr_list.append(protein2reaction_repr.detach().cpu().numpy())
        protein_seq_list.extend(protein_seq)
        text_seq_list.extend(text_seq)
        reaction_seq_list.extend(reaction_seq)

    protein_repr_array = np.concatenate(protein_repr_list)
    text_repr_array= np.concatenate(text_repr_list)
    reaction_repr_array = np.concatenate(reaction_repr_list)
    reaction2protein_repr_array = np.concatenate(reaction2protein_repr_list)
    protein2reaction_repr_array = np.concatenate(protein2reaction_repr_list)
    return protein_repr_array, text_repr_array, reaction_repr_array, reaction2protein_repr_array, protein2reaction_repr_array,protein_seq_list, text_seq_list, reaction_seq_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--protein_backbone_model", type=str, default="ProtT5", choices=["ProtBERT", "ProtBERT_BFD", "ProtT5"])
    parser.add_argument("--text_backbone_model", type=str, default="SciBERT")
    parser.add_argument("--reaction_backbone_model", type=str, default="rxnfp")
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    parser.add_argument("--reaction_max_sequence_len", type=int, default=512)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=True)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.set_defaults(use_AMP=True)

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

    ##### Load pretrained protein model
    if args.protein_backbone_model == "ProtBERT":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir="../../data/temp_pretrained_ProtBert")
    elif args.protein_backbone_model == "ProtBERT_BFD":
        protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert_BFD")
        protein_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="../../data/temp_pretrained_ProtBert_BFD")
    elif args.protein_backbone_model == "ProtT5":
        protein_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, cache_dir="../../data/temp_pretrained_prot_t5_xl_half_uniref50-enc")
        protein_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", cache_dir="../../data/temp_pretrained_prot_t5_xl_half_uniref50-enc")
    protein_dim = 1024
    input_model_path = os.path.join(args.pretrained_folder, "protein_model.pth")
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein_model.load_state_dict(state_dict)

    ##### Load pretrained text model
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    text_dim  = 768
    input_model_path = os.path.join(args.pretrained_folder, "text_model.pth")
    print("Loading text model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict)

    #### Load pretrained reaction model
    reaction_tokenizer = SmilesTokenizer.from_pretrained("/disk1/jyang4/repos/rxnfp/rxnfp/models/transformers/bert_pretrained/vocab.txt")
    reaction_model = BertModel.from_pretrained("/disk1/jyang4/repos/rxnfp/rxnfp/models/transformers/bert_pretrained")
    reaction_dim = 256
    input_model_path = os.path.join(args.pretrained_folder, "reaction_model.pth")
    print("Loading reaction model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    reaction_model.load_state_dict(state_dict)

    ##### Load pretrained protein2latent model
    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "protein2latent_model.pth")
    print("Loading protein2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    protein2latent_model.load_state_dict(state_dict)

    ##### Load pretrained text2latent model
    text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "text2latent_model.pth")
    print("Loading text2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent_model.load_state_dict(state_dict)

    ##### Load pretrained reaction2latent model
    reaction2latent_model = nn.Linear(reaction_dim, args.SSL_emb_dim)
    input_model_path = os.path.join(args.pretrained_folder, "reaction2latent_model.pth")
    print("Loading reaction2latent model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    reaction2latent_model.load_state_dict(state_dict)

    #### Load pretrained reaction2protein_facilitator model
    reaction2protein_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    state_dict = torch.load(os.path.join(args.pretrained_folder, "reaction2protein_facilitator_model.pth"), map_location='cpu')
    reaction2protein_facilitator_model.load_state_dict(state_dict)

    #### Load pretrained protein2reaction_facilitator model
    protein2reaction_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    state_dict = torch.load(os.path.join(args.pretrained_folder, "protein2reaction_facilitator_model.pth"), map_location='cpu')
    protein2reaction_facilitator_model.load_state_dict(state_dict)

    model = CREEPModel(protein_model, text_model, reaction_model, protein2latent_model, text2latent_model, reaction2latent_model, reaction2protein_facilitator_model, protein2reaction_facilitator_model, args.protein_backbone_model, args.text_backbone_model, args.reaction_backbone_model)
    model.eval()
    model.to(device)

    path="../../data/PECT/test_sets/" + args.dataset + ".csv"

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

    dataset = CREEPDataset(
        path=path,
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len,
        reaction_tokenizer=reaction_tokenizer,
        reaction_max_sequence_len=args.reaction_max_sequence_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    protein_repr_array, text_repr_array, reaction_repr_array, reaction2protein_repr_array, protein2reaction_repr_array, protein_seq_list, text_seq_list, reaction_seq_list = extract(dataloader, AMP=args.use_AMP)

    assert args.pretrained_folder is not None
    output_folder = os.path.join(args.pretrained_folder, "step_02_extract_representation")
    os.makedirs(output_folder, exist_ok=True)

    saved_file_path = os.path.join(output_folder, args.dataset + "_representations")
    #TODO: don't save all the tensors if you don't need them all
    np.savez(saved_file_path, protein_repr_array=protein_repr_array, text_repr_array=text_repr_array, reaction_repr_array=reaction_repr_array, reaction2protein_repr_array=reaction2protein_repr_array, protein2reaction_repr_array=protein2reaction_repr_array)

    df = pd.DataFrame({"protein_sequence": protein_seq_list, "text_sequence": text_seq_list, "reaction_sequence": reaction_seq_list})
    #don't save for now
    #df.to_csv(os.path.join(output_folder, "pairs.tsv"), index=False, sep="\t")
