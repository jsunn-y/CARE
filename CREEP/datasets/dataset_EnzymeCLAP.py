import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

def encode_sequence(sequence, tokenizer, max_sequence_len):    
    sequence_encode = tokenizer(sequence, truncation=True, max_length=max_sequence_len, padding='max_length', return_tensors='pt')
    input_ids = sequence_encode.input_ids.squeeze()
    attention_mask = sequence_encode.attention_mask.squeeze()
    return input_ids, attention_mask

class EnzymeCLAPDataset(Dataset):
    def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len):
        self.path = path
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len

        self.df = pd.read_csv(path).reset_index()
        #df.dropna(subset='reaction_smiles', inplace=True)
        #df['reaction_smiles'] = df['reaction_smiles'].str.split('|').str[0]
        #df['activity'] = df['activity'].str.split('|').str[0]
        #df['activity'] = df['activity'].str.replace('"', '')
        # df['reaction_eq'] = df['reaction_eq'].str.split('|').str[0]
        # df['reaction_eq'] = df['reaction_eq'].str.replace('"', '')
        #df['sequence'] =  df['sequence'].str.replace('"', '')

        self.protein_sequence_list = self.df['sequence'].values.tolist()
        self.protein_sequence_list = [" ".join(protein_sequence) for protein_sequence in self.protein_sequence_list]
        self.df['sequence'] = self.protein_sequence_list
        self.text_sequence_list = self.df['reaction_eq'].values.tolist()
        self.reaction_sequence_list = self.df['reaction_smiles'].values.tolist()

        print("num of (protein-sequence, text, reaction) triplets: {}".format(len(self.protein_sequence_list)))
        print(self.protein_sequence_list[0])
        print(self.text_sequence_list[0])
        print(self.reaction_sequence_list[0])

        return

    def __getitem__(self, index):
        protein_sequence = self.protein_sequence_list[index]
        text_sequence = self.text_sequence_list[index]
        reaction_sequence = self.reaction_sequence_list[index]

        protein_sequence_input_ids, protein_sequence_attention_mask = encode_sequence(protein_sequence, self.protein_tokenizer, self.protein_max_sequence_len)
        text_sequence_input_ids, text_sequence_attention_mask = encode_sequence(text_sequence, self.text_tokenizer, self.text_max_sequence_len)
        reaction_sequence_input_ids, reaction_sequence_attention_mask = encode_sequence(reaction_sequence, self.reaction_tokenizer, self.reaction_max_sequence_len)
        
        batch = {
            "protein_sequence": protein_sequence,
            "protein_sequence_input_ids": protein_sequence_input_ids,
            "protein_sequence_attention_mask": protein_sequence_attention_mask,
            "text_sequence": text_sequence,
            "text_sequence_input_ids": text_sequence_input_ids,
            "text_sequence_attention_mask": text_sequence_attention_mask,
            "reaction_sequence": reaction_sequence,
            "reaction_sequence_input_ids": reaction_sequence_input_ids,
            "reaction_sequence_attention_mask": reaction_sequence_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.protein_sequence_list)

def mine_negative(anchor_protein_sequence, protein_sequence2ec, ec2protein_sequence):
    anchor_ec = protein_sequence2ec[anchor_protein_sequence]
    pos_ec = random.choice(anchor_ec)
    #get the other ECs (different from how they do it in CLEAN)
    other_ecs = list(ec2protein_sequence.keys())
    other_ecs.remove(pos_ec)
    neg_ec = random.choice(other_ecs)
    # print('Neg:')
    # print(neg_ec)
    neg_protein_sequence = random.choice(ec2protein_sequence[neg_ec])
    return neg_protein_sequence

def mine_positive(anchor_protein_sequence, protein_sequence2ec, ec2protein_sequence):
    pos_ec = random.choice(protein_sequence2ec[anchor_protein_sequence])
    pos_protein_sequence = anchor_protein_sequence
    if len(ec2protein_sequence[pos_ec]) == 1:
        return pos_protein_sequence + '_' + str(random.randint(0, 9))
    while pos_protein_sequence == anchor_protein_sequence:
        pos_protein_sequence = random.choice(ec2protein_sequence[pos_ec])
    return pos_protein_sequence

class EnzymeCLAPDataset_with_mine_batch(Dataset):
    """
    Dataset is the length of the number of unique EC numbers.
    Use a batch size of 1 here and n_neg to determine the number of negative examples to include in the batch.
    Ensures that the rest of the batch is negative protein examples.
    """
    def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len, n_neg, loop_over_unique_EC=False):
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len

        self.df = pd.read_csv(path).reset_index()

        self.protein_sequence_list = self.df['sequence'].values.tolist()
        self.protein_sequence_list = [" ".join(protein_sequence) for protein_sequence in self.protein_sequence_list]
        self.df['sequence'] = self.protein_sequence_list
        self.text_sequence_list = self.df['name'].values.tolist()
        #self.reaction_sequence_list = self.df['reaction_smiles'].values.tolist()

        print("num of protein_sequences: {}".format(len(self.protein_sequence_list)))
        print(self.protein_sequence_list[0])
        print(self.text_sequence_list[0])
        
        self.df['brenda'] = self.df['brenda'].str.split('.').str[:4].str.join('.') #number here is the number of levels of EC
        #dictionary mapping from EC to smiles and vice versa
        self.df['index'] = self.df.index
        self.ec2index = self.df.groupby('brenda')['index'].apply(list).to_frame().to_dict()['index']
        self.ec2rxns = np.load('/disk1/jyang4/repos/ProteinDT_submission/data/SwissProtEnzymeCLAP/processed_data/EC2rxns_ECreact.npy', allow_pickle=True).item()

        # #print(len(ec_smiles['1.1']))
        # self.protein_sequence2ec = self.df.groupby('sequence')['brenda'].apply(list).to_frame().to_dict()['brenda']

        self.n_negs = n_neg
        self.loop_over_unique_ec = loop_over_unique_EC
        #loop by unique ECs
        if loop_over_unique_EC:
            self.full_list = []
            for ec in self.ec2index.keys(): #alternatively loop through all unique ECs instead of protein sequences
                if '-' not in ec:
                    self.full_list.append(ec)
        #otherwise loop over every protein sequence
        else:
            self.full_list = list(range(len(self.protein_sequence_list)))

        return

    def __getitem__(self, index):

        if self.loop_over_unique_ec:
            anchor_ec = self.full_list[index]
            anchor_options = self.ec2index[anchor_ec]
            anchor_idx = random.choice(anchor_options) #entry index in the database
        else:
            anchor_idx = index
            anchor_ec = self.df.loc[anchor_idx, 'brenda'] #might be slow

        #choose randomly based on either choosing from (1) protein 
        #TODO: ensure that there are no overlaps in EC
        # negative_options = self.df[self.df['brenda'] != anchor_ec].index #this might be quite slow
        # negative_idxs = random.sample(list(negative_options), self.n_negs)

        #alternatively choose from (2) EC numbers without replacement
        other_ecs = list(self.ec2index.keys())
        other_ecs.remove(anchor_ec)
        negative_ecs = random.sample(other_ecs, self.n_negs) #sample EC numbers without replacement
        
        idxs = [anchor_idx]
        for ec in negative_ecs:
            idxs.append(random.choice(self.ec2index[ec])) #sample an index from each EC number

        all_protein_sequence_input_ids = torch.zeros(len(idxs), self.protein_max_sequence_len, dtype=torch.long)
        all_protein_sequence_attention_mask = torch.zeros(len(idxs), self.protein_max_sequence_len, dtype=torch.long)
        all_text_sequence_input_ids = torch.zeros(len(idxs), self.text_max_sequence_len, dtype=torch.long)
        all_text_sequence_attention_mask = torch.zeros(len(idxs), self.text_max_sequence_len, dtype=torch.long)
        all_reaction_sequence_input_ids = torch.zeros(len(idxs), self.reaction_max_sequence_len, dtype=torch.long)
        all_reaction_sequence_attention_mask = torch.zeros(len(idxs), self.reaction_max_sequence_len, dtype=torch.long)

        for i, idx in enumerate(idxs):
            protein_sequence = self.protein_sequence_list[idx]
            text_sequence = self.text_sequence_list[idx]
            reaction_sequence = random.choice(self.ec2rxns[anchor_ec])

            all_protein_sequence_input_ids[i,:], all_protein_sequence_attention_mask[i,:] = encode_sequence( protein_sequence, self.protein_tokenizer, self.protein_max_sequence_len)
            all_text_sequence_input_ids[i,:], all_text_sequence_attention_mask[i,:] = encode_sequence( text_sequence, self.text_tokenizer, self.text_max_sequence_len)
            all_reaction_sequence_input_ids[i,:], all_reaction_sequence_attention_mask[i,:] = encode_sequence( reaction_sequence, self.reaction_tokenizer, self.reaction_max_sequence_len)
        
        batch = {
            "protein_sequence_input_ids": all_protein_sequence_input_ids,
            "protein_sequence_attention_mask": all_protein_sequence_attention_mask,
            "text_sequence_input_ids": all_text_sequence_input_ids,
            "text_sequence_attention_mask": all_text_sequence_attention_mask,
            "reaction_sequence_input_ids": all_reaction_sequence_input_ids,
            "reaction_sequence_attention_mask": all_reaction_sequence_attention_mask,
        }
        return batch

    
    def __len__(self):
        return len(self.full_list)
    
class EnzymeCLAPDataset_with_mine_EC(EnzymeCLAPDataset):
    """
    Dataset is the length of the number of unique EC numbers.
    Ensures that a certiain number of additional positive and negative protein examples are included in the batch.
    A bit convoluted - used to enforce protein embedding patterns like in CLEAN.
    """
    def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len, n_pos, n_neg):
        super().__init__(path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len)
        
        self.df['brenda'] = self.df['brenda'].str.split('.').str[:4].str.join('.') #number here is the number of levels of EC
        #dictionary mapping from EC to smiles and vice versa
        self.ec2protein_sequence = self.df.groupby('brenda')['sequence'].apply(list).to_frame().to_dict()['sequence']
        #print(len(ec_smiles['1.1']))
        self.protein_sequence2ec = self.df.groupby('sequence')['brenda'].apply(list).to_frame().to_dict()['brenda']

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        for ec in self.ec2protein_sequence.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        return

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor_protein_sequence = random.choice(self.ec2protein_sequence[anchor_ec])

        #this could slow things down (better to zip together the idx and the correponding protein sequence)
        temp = self.df[self.df['sequence'] == anchor_protein_sequence]
        anchor_idx = temp[temp['brenda'] == anchor_ec].index[0]

        anchor_text_sequence = self.text_sequence_list[anchor_idx]
        anchor_reaction_sequence = self.reaction_sequence_list[anchor_idx]

        all_protein_sequence_input_ids, all_protein_sequence_attention_mask = encode_sequence(anchor_protein_sequence, self.protein_tokenizer, self.protein_max_sequence_len)
        anchor_text_sequence_input_ids, anchor_text_sequence_attention_mask = encode_sequence(anchor_text_sequence, self.text_tokenizer, self.text_max_sequence_len)
        anchor_reaction_sequence_input_ids, anchor_reaction_sequence_attention_mask = encode_sequence(anchor_reaction_sequence, self.reaction_tokenizer, self.reaction_max_sequence_len)
        
        # pos_protein_sequence_input_ids = torch.zeros(self.n_pos, self.reaction_max_sequence_len)
        # pos_protein_sequence_attention_mask = torch.zeros(self.n_pos, self.reaction_max_sequence_len)
        # neg_protein_sequence_input_ids = torch.zeros(self.n_neg, self.reaction_max_sequence_len)
        # neg_protein_sequence_attention_mask = torch.zeros(self.n_neg, self.reaction_max_sequence_len)

        for _ in range(self.n_pos):
            pos_protein_sequence = mine_positive(anchor_protein_sequence, self.protein_sequence2ec, self.ec2protein_sequence)
            input_ids, attention_mask = encode_sequence(pos_protein_sequence, self.reaction_tokenizer, self.protein_max_sequence_len)
            all_protein_sequence_input_ids = torch.cat((all_protein_sequence_input_ids, input_ids), 0)
            all_protein_sequence_attention_mask = torch.cat((all_protein_sequence_attention_mask, attention_mask), 0)
        for _ in range(self.n_neg):
            neg_protein_sequence = mine_negative(anchor_protein_sequence, self.protein_sequence2ec, self.ec2protein_sequence)
            input_ids, attention_mask = encode_sequence(neg_protein_sequence, self.reaction_tokenizer, self.protein_max_sequence_len)
            all_protein_sequence_input_ids = torch.cat((all_protein_sequence_input_ids, input_ids), 0)
            all_protein_sequence_attention_mask = torch.cat((all_protein_sequence_attention_mask, attention_mask), 0)
        
        batch = {
            "all_protein_sequence_input_ids": all_protein_sequence_input_ids,
            "all_protein_sequence_attention_mask": all_protein_sequence_attention_mask,
            "text_sequence_input_ids": anchor_text_sequence_input_ids,
            "text_sequence_attention_mask": anchor_text_sequence_attention_mask,
            "reaction_sequence_input_ids": anchor_reaction_sequence_input_ids,
            "reaction_sequence_attention_mask": anchor_reaction_sequence_attention_mask,
        }
        return batch

    
    def __len__(self):
        return len(self.full_list)


#needs to be updated to be used
# if __name__ == "__main__":
#     from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
#     from torch.utils.data import DataLoader

#     protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_ProtBert")

#     text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir="../../data/temp_pretrained_SciBert")

#     dataset = EnzymeCLAPDataset(
#         root="../../data/EnzymeCLAP",
#         protein_tokenizer=protein_tokenizer,
#         text_tokenizer=text_tokenizer,
#         protein_max_sequence_len=512,
#         text_max_sequence_len=512
#     )
#     print("len of dataset", len(dataset))
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
#     for batch in dataloader:
#         protein_sequence_list = batch["protein_sequence"]
#         text_sequence_list = batch["text_sequence"] 
#         reaction_sequence_list = batch["reaction_sequence"] 
#         for protein_sequence, text_sequence, reaction_sequence in zip(protein_sequence_list, text_sequence_list, reaction_sequence_list):
#             print(protein_sequence.replace(" ", ""))
#             print(text_sequence)
#             print(reaction_sequence)
#             print()
#         break
