import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel#, BertForMaskedLM
from torch.utils.data import DataLoader

from ProteinDT.models import CREEPModel, AEFacilitatorModel
from ProteinDT.datasets import CREEPDatasetMineBatch
from ProteinDT.utils.tokenization import SmilesTokenizer
import sys

#make a logger to save save everything from the terminal to a txt
class Logger:
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.CL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def save_model(save_best):
    if args.output_model_dir is None:
        return
    
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
        
        saved_file_path = os.path.join(args.output_model_dir, "text_{}".format(model_file))
        torch.save(text_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein_{}".format(model_file))
        torch.save(protein_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction_{}".format(model_file))
        torch.save(reaction_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "text2latent_{}".format(model_file))
        torch.save(text2latent_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein2latent_{}".format(model_file))
        torch.save(protein2latent_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction2latent_{}".format(model_file))
        torch.save(reaction2latent_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction2protein_facilitator_{}".format(model_file))
        torch.save(reaction2protein_facilitator_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "protein2reaction_facilitator_{}".format(model_file))
        torch.save(protein2reaction_facilitator_model.state_dict(), saved_file_path)

    else:
        model_file = "model_final.pth"

        saved_file_path = os.path.join(args.output_model_dir, "text_{}".format(model_file))
        torch.save(text_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein_{}".format(model_file))
        torch.save(protein_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction_{}".format(model_file))
        torch.save(reaction_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "text2latent_{}".format(model_file))
        torch.save(text2latent_model.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "protein2latent_{}".format(model_file))
        torch.save(protein2latent_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction2latent_{}".format(model_file))
        torch.save(reaction2latent_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "reaction2protein_facilitator_{}".format(model_file))
        torch.save(reaction2protein_facilitator_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "protein2reaction_facilitator_{}".format(model_file))
        torch.save(protein2reaction_facilitator_model.state_dict(), saved_file_path)

    return

def train_AMP(dataloader):
    scaler = torch.cuda.amp.GradScaler()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc, accum_contrastive_loss, accum_generative_loss = 0, 0, 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence_input_ids = batch["protein_sequence_input_ids"].to(device)
        protein_sequence_attention_mask = batch["protein_sequence_attention_mask"].to(device)
        text_sequence_input_ids = batch["text_sequence_input_ids"].to(device)
        text_sequence_attention_mask = batch["text_sequence_attention_mask"].to(device)
        reaction_sequence_input_ids = batch["reaction_sequence_input_ids"].to(device)
        reaction_sequence_attention_mask = batch["reaction_sequence_attention_mask"].to(device)

        if args.mine_batch:
            protein_sequence_input_ids = torch.squeeze(protein_sequence_input_ids)
            protein_sequence_attention_mask = torch.squeeze(protein_sequence_attention_mask)
            text_sequence_input_ids = torch.squeeze(text_sequence_input_ids)
            text_sequence_attention_mask = torch.squeeze(text_sequence_attention_mask)
            reaction_sequence_input_ids = torch.squeeze(reaction_sequence_input_ids)
            reaction_sequence_attention_mask = torch.squeeze(reaction_sequence_attention_mask)
        
        with torch.cuda.amp.autocast():
            protein_repr, text_repr, reaction_repr, reaction2protein_repr, protein2reaction_repr = model(protein_sequence_input_ids, protein_sequence_attention_mask, text_sequence_input_ids, text_sequence_attention_mask, reaction_sequence_input_ids, reaction_sequence_attention_mask)

            if args.use_three_modalities:
                loss_01, acc_01 = do_CL(protein_repr, text_repr, args)
                loss_02, acc_02 = do_CL(text_repr, protein_repr, args)
                loss_03, acc_03 = do_CL(text_repr, reaction_repr, args)
                loss_04, acc_04 = do_CL(protein_repr, reaction_repr, args)
                loss_05, acc_05 = do_CL(reaction_repr, text_repr, args)
                loss_06, acc_06 = do_CL(reaction_repr, protein_repr, args)

                contrastive_loss = (loss_01 + loss_02 + loss_03 + loss_04 + loss_05 + loss_06) / 6
                contrastive_acc = (acc_01 + acc_02 + acc_03 + acc_04 + acc_05 + acc_06) / 6
            else:
                #manually train with just two modalities
                loss_01, acc_01 = do_CL(protein_repr, reaction_repr, args)
                loss_02, acc_02 = do_CL(reaction_repr, protein_repr, args)

                contrastive_loss = (loss_01 + loss_02) / 2
                contrastive_acc = (acc_01 + acc_02) / 2
            
            #print(contrastive_loss)
            criterion = nn.MSELoss()
            # print(reaction2protein_repr.shape, protein_repr.shape)
            generative_loss = criterion(reaction2protein_repr, protein_repr) + criterion(protein2reaction_repr, reaction_repr)
            #print(reaction2protein_loss, protein2reaction_loss)
            # generative_loss = reaction2protein_loss + protein2reaction_loss
            #print(generative_loss)
            loss = args.alpha_contrastive*contrastive_loss + args.alpha_generative*generative_loss
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_contrastive_loss += contrastive_loss.item()
        accum_generative_loss += generative_loss.item()
        accum_loss += loss.item()
        accum_acc += contrastive_acc
        if args.verbose and batch_idx % 100 == 0:
            print(loss.item(), contrastive_acc)
        
    accum_loss /= len(L)
    accum_contrastive_loss /= len(L)
    accum_generative_loss /= len(L)
    accum_acc /= len(L)
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tGenerative Loss: {:.5f}\tTotal Loss: {:.5f}\tTime: {:.5f}".format(accum_contrastive_loss, accum_acc, accum_generative_loss, accum_loss,time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mine_batch", dest="mine_batch", action="store_true") #chooses a batch such that all the EC numbers are different iwithin the batch
    parser.set_defaults(mine_batch=True)
    parser.add_argument("--loop_over_unique_EC", dest="loop_over_unique_EC", action="store_true") #loops through unique EC numbers rather than protein entries
    parser.set_defaults(loop_over_unique_EC=True)

    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='CLEANsplit100.csv')
    parser.add_argument("--protein_backbone_model", type=str, default="ProtT5", choices=["ProtBERT", "ProtBERT_BFD", "ProtT5"])
    parser.add_argument("--text_backbone_model", type=str, default="SciBERT")
    parser.add_argument("--reaction_backbone_model", type=str, default="rxnfp")
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--text_max_sequence_len", type=int, default=512)
    parser.add_argument("--reaction_max_sequence_len", type=int, default=512)
    parser.add_argument("--protein_lr", type=float, default=1e-5)
    parser.add_argument("--protein_lr_scale", type=float, default=1)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1e-1)
    parser.add_argument("--reaction_lr", type=float, default=1e-5)
    parser.add_argument("--reaction_lr_scale", type=float, default=1)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--CL_loss", type=str, default="EBM_NCE")
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--use_three_modalities", dest="use_three_modalities", action="store_true")
    parser.add_argument("--use_two_modalities", dest="use_three_modalities", action="store_false")
    parser.set_defaults(use_three_modalities=True)

    parser.add_argument("--alpha_contrastive", type=float, default=1)
    parser.add_argument("--alpha_generative", type=float, default=0)

    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=False)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    
    parser.add_argument("--use_AMP", dest="use_AMP", action="store_true")
    parser.add_argument("--no_AMP", dest="use_AMP", action="store_false")
    parser.add_argument("--load_pretrained", dest="load_pretrained", action="store_true")
    parser.set_defaults(use_AMP=True)
    parser.set_defaults(load_pretrained=False)

    parser.add_argument("--output_model_dir", type=str, default=None)

    args = parser.parse_args()

    log_file = "{}/log.txt".format(args.output_model_dir)
    #make log file if it doesn't exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    sys.stdout = Logger(log_file, 'w')

    print("arguments", args)
    #save the args to the output folder
    # if args.output_model_dir is not None:
    #     os.makedirs(args.output_model_dir, exist_ok=True)
    #     with open(os.path.join(args.output_model_dir, "config.txt"), "w") as f:
    #         f.write(str(args))

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    if args.load_pretrained:
        state_dict = torch.load('/disk1/jyang4/repos/ProteinDT_submission/output/ProteinDT/ProtT5_encoder/protein_model.pth', map_location='cpu')
        protein_model.load_state_dict(state_dict)


    # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")
    if args.load_pretrained:
        state_dict = torch.load('/disk1/jyang4/repos/ProteinDT_submission/output/ProteinDT/ProtT5_encoder/text_model.pth', map_location='cpu')
        text_model.load_state_dict(state_dict)
    text_dim  = 768
    
    #reaction embeddings
    #consider using the finetuning the model for EC classification beforehand
    reaction_tokenizer = SmilesTokenizer.from_pretrained("/disk1/jyang4/repos/rxnfp/rxnfp/models/transformers/bert_pretrained/vocab.txt")
    reaction_model = BertModel.from_pretrained("/disk1/jyang4/repos/rxnfp/rxnfp/models/transformers/bert_pretrained")
    load_CLEANed = True
    if args.load_pretrained:
        state_dict = torch.load('/disk1/jyang4/repos/ProteinDT_submission/examples/finetuning_rxnfp/finetuned_level1_epoch1_level2_epoch3.pth', map_location='cpu')
        reaction_model.load_state_dict(state_dict)
    reaction_dim = 256

    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    if args.load_pretrained:
        state_dict = torch.load('/disk1/jyang4/repos/ProteinDT_submission/output/ProteinDT/ProtT5_encoder/protein2latent_model.pth', map_location='cpu')
        protein2latent_model.load_state_dict(state_dict)
    text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
    if args.load_pretrained:
        state_dict = torch.load('/disk1/jyang4/repos/ProteinDT_submission/output/ProteinDT/ProtT5_encoder/text2latent_model.pth', map_location='cpu')
        text2latent_model.load_state_dict(state_dict)
    reaction2latent_model = nn.Linear(reaction_dim, args.SSL_emb_dim)

    reaction2protein_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    protein2reaction_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)

    #bring it all together
    model = CREEPModel(protein_model, text_model, reaction_model, protein2latent_model, text2latent_model, reaction2latent_model, reaction2protein_facilitator_model, protein2reaction_facilitator_model, args.protein_backbone_model, args.text_backbone_model, args.reaction_backbone_model)

    if torch.cuda.device_count() > 1:
        # parallel models
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        neo_batch_size = args.batch_size * torch.cuda.device_count()
        print("batch size from {} to {}".format(args.batch_size, neo_batch_size))
        args.batch_size = neo_batch_size
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model_param_group = [
        {"params": protein_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": text_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {"params": reaction_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": protein2latent_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": text2latent_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {"params": reaction2latent_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": reaction2protein_facilitator_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": protein2reaction_facilitator_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    assert args.mine_batch

    dataset = CREEPDatasetMineBatch(
    path="../../data/PECT/" + args.dataset,
    protein_tokenizer=protein_tokenizer,
    text_tokenizer=text_tokenizer,
    protein_max_sequence_len=args.protein_max_sequence_len,
    text_max_sequence_len=args.text_max_sequence_len,
    reaction_tokenizer=reaction_tokenizer,
    reaction_max_sequence_len=args.reaction_max_sequence_len,
    n_neg=args.batch_size - 1,
    loop_over_unique_EC=args.loop_over_unique_EC
    )
    args.batch_size = 1

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #output_folder = os.path.join(args.output_model_dir)

    if args.loop_over_unique_EC:
        args.epochs = 50 #14 is approximately how many would be the same number of samples as looping through sequences

    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        assert args.use_AMP
        train_AMP(dataloader)