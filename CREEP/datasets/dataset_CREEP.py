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

# class CREEPDatasetOld(Dataset):
#     """
#     Loops through triplets of (protein-sequence, text, reaction) from test datasets.
#     Used for extraction of representations.
#     """
#     def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len):
#         self.path = path
#         self.protein_tokenizer = protein_tokenizer
#         self.text_tokenizer = text_tokenizer
#         self.reaction_tokenizer = reaction_tokenizer
#         self.protein_max_sequence_len = protein_max_sequence_len
#         self.text_max_sequence_len = text_max_sequence_len
#         self.reaction_max_sequence_len = reaction_max_sequence_len

#         self.df = pd.read_csv(path).reset_index()
#         #df.dropna(subset='reaction_smiles', inplace=True)
#         #df['reaction_smiles'] = df['reaction_smiles'].str.split('|').str[0]
#         #df['activity'] = df['activity'].str.split('|').str[0]
#         #df['activity'] = df['activity'].str.replace('"', '')
#         # df['reaction_eq'] = df['reaction_eq'].str.split('|').str[0]
#         # df['reaction_eq'] = df['reaction_eq'].str.replace('"', '')
#         #df['sequence'] =  df['sequence'].str.replace('"', '')

#         self.protein_sequence_list = self.df['sequence'].values.tolist()
#         self.protein_sequence_list = [" ".join(protein_sequence) for protein_sequence in self.protein_sequence_list]
#         self.df['sequence'] = self.protein_sequence_list
#         self.text_sequence_list = self.df['name'].values.tolist()
#         self.reaction_sequence_list = self.df['reaction_smiles'].values.tolist()

#         print("num of (protein-sequence, text, reaction) triplets: {}".format(len(self.protein_sequence_list)))
#         print(self.protein_sequence_list[0])
#         print(self.text_sequence_list[0])
#         print(self.reaction_sequence_list[0])

#         return

#     def __getitem__(self, index):
#         protein_sequence = self.protein_sequence_list[index]
#         text_sequence = self.text_sequence_list[index]
#         reaction_sequence = self.reaction_sequence_list[index]

#         protein_sequence_input_ids, protein_sequence_attention_mask = encode_sequence(protein_sequence, self.protein_tokenizer, self.protein_max_sequence_len)
#         text_sequence_input_ids, text_sequence_attention_mask = encode_sequence(text_sequence, self.text_tokenizer, self.text_max_sequence_len)
#         reaction_sequence_input_ids, reaction_sequence_attention_mask = encode_sequence(reaction_sequence, self.reaction_tokenizer, self.reaction_max_sequence_len)
        
#         batch = {
#             "protein_sequence": protein_sequence,
#             "protein_sequence_input_ids": protein_sequence_input_ids,
#             "protein_sequence_attention_mask": protein_sequence_attention_mask,
#             "text_sequence": text_sequence,
#             "text_sequence_input_ids": text_sequence_input_ids,
#             "text_sequence_attention_mask": text_sequence_attention_mask,
#             "reaction_sequence": reaction_sequence,
#             "reaction_sequence_input_ids": reaction_sequence_input_ids,
#             "reaction_sequence_attention_mask": reaction_sequence_attention_mask,
#         }

#         return batch
    
#     def __len__(self):
#         return len(self.protein_sequence_list)
    
# class CREEPDatasetMineBatchOld(Dataset):
#     """
#     Dataset is the length of the number of unique EC numbers.
#     Use a batch size of 1 here and n_neg to determine the number of negative examples to include in the batch.
#     Ensures that the rest of the batch is negative protein examples.
#     Used for training
#     """
#     def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len, n_neg, loop_over_unique_EC=False):
#         self.protein_tokenizer = protein_tokenizer
#         self.text_tokenizer = text_tokenizer
#         self.reaction_tokenizer = reaction_tokenizer
#         self.protein_max_sequence_len = protein_max_sequence_len
#         self.text_max_sequence_len = text_max_sequence_len
#         self.reaction_max_sequence_len = reaction_max_sequence_len

#         self.df = pd.read_csv(path).reset_index()

#         self.protein_sequence_list = self.df['sequence'].values.tolist()
#         self.protein_sequence_list = [" ".join(protein_sequence) for protein_sequence in self.protein_sequence_list]
#         self.df['sequence'] = self.protein_sequence_list
#         self.text_sequence_list = self.df['name'].values.tolist()
#         #self.reaction_sequence_list = self.df['reaction_smiles'].values.tolist()

#         print("num of protein-EC pairs: {}".format(len(self.protein_sequence_list)))
        
#         print(self.protein_sequence_list[0])
#         print(self.text_sequence_list[0])
        
#         self.df['brenda'] = self.df['brenda'].str.split('.').str[:4].str.join('.') #number here is the number of levels of EC
#         #dictionary mapping from EC to smiles and vice versa
#         self.df['index'] = self.df.index
#         self.ec2index = self.df.groupby('brenda')['index'].apply(list).to_frame().to_dict()['index']
#         self.ec2rxns = np.load('../../data/PECT/EC2rxns_train.npy', allow_pickle=True).item()

#         print("num of ECs: {}".format(len(self.ec2rxns)))
#         print("num of reactions: {}".format(sum([len(self.ec2rxns[ec]) for ec in self.ec2rxns.keys()])))

#         # #print(len(ec_smiles['1.1']))
#         # self.protein_sequence2ec = self.df.groupby('sequence')['brenda'].apply(list).to_frame().to_dict()['brenda']

#         self.n_negs = n_neg
#         self.loop_over_unique_ec = loop_over_unique_EC
#         #loop by unique ECs
#         if loop_over_unique_EC:
#             self.full_list = []
#             for ec in self.ec2index.keys(): #alternatively loop through all unique ECs instead of protein sequences
#                 if '-' not in ec:
#                     self.full_list.append(ec)
#         #otherwise loop over every protein sequence
#         else:
#             self.full_list = list(range(len(self.protein_sequence_list)))
#         ####TODO: loop over reactions instead of EC numbers####

#         return

#     def __getitem__(self, index):

#         if self.loop_over_unique_ec:
#             anchor_ec = self.full_list[index]
#             anchor_options = self.ec2index[anchor_ec]
#             anchor_idx = random.choice(anchor_options) #entry index in the database
#         else:
#             anchor_idx = index
#             anchor_ec = self.df.loc[anchor_idx, 'brenda'] #might be slow

#         #choose randomly based on either choosing from (1) protein 
#         #TODO: ensure that there are no overlaps in EC
#         # negative_options = self.df[self.df['brenda'] != anchor_ec].index #this might be quite slow
#         # negative_idxs = random.sample(list(negative_options), self.n_negs)

#         #alternatively choose from (2) EC numbers without replacement
#         other_ecs = list(self.ec2index.keys())
#         other_ecs.remove(anchor_ec)
#         negative_ecs = random.sample(other_ecs, self.n_negs) #sample EC numbers without replacement
        
#         idxs = [anchor_idx]
#         ecs = [anchor_ec]
#         for ec in negative_ecs:
#             idxs.append(random.choice(self.ec2index[ec])) #sample an index from each EC number
#             ecs.append(ec)

#         all_protein_sequence_input_ids = torch.zeros(len(idxs), self.protein_max_sequence_len, dtype=torch.long)
#         all_protein_sequence_attention_mask = torch.zeros(len(idxs), self.protein_max_sequence_len, dtype=torch.long)
#         all_text_sequence_input_ids = torch.zeros(len(idxs), self.text_max_sequence_len, dtype=torch.long)
#         all_text_sequence_attention_mask = torch.zeros(len(idxs), self.text_max_sequence_len, dtype=torch.long)
#         all_reaction_sequence_input_ids = torch.zeros(len(idxs), self.reaction_max_sequence_len, dtype=torch.long)
#         all_reaction_sequence_attention_mask = torch.zeros(len(idxs), self.reaction_max_sequence_len, dtype=torch.long)

#         for i, (idx, ec) in enumerate(zip(idxs, ecs)):
#             protein_sequence = self.protein_sequence_list[idx]
#             text_sequence = self.text_sequence_list[idx]
#             reaction_sequence = random.choice(self.ec2rxns[ec])

#             all_protein_sequence_input_ids[i,:], all_protein_sequence_attention_mask[i,:] = encode_sequence( protein_sequence, self.protein_tokenizer, self.protein_max_sequence_len)
#             all_text_sequence_input_ids[i,:], all_text_sequence_attention_mask[i,:] = encode_sequence( text_sequence, self.text_tokenizer, self.text_max_sequence_len)
#             all_reaction_sequence_input_ids[i,:], all_reaction_sequence_attention_mask[i,:] = encode_sequence( reaction_sequence, self.reaction_tokenizer, self.reaction_max_sequence_len)
        
#         batch = {
#             "protein_sequence_input_ids": all_protein_sequence_input_ids,
#             "protein_sequence_attention_mask": all_protein_sequence_attention_mask,
#             "text_sequence_input_ids": all_text_sequence_input_ids,
#             "text_sequence_attention_mask": all_text_sequence_attention_mask,
#             "reaction_sequence_input_ids": all_reaction_sequence_input_ids,
#             "reaction_sequence_attention_mask": all_reaction_sequence_attention_mask,
#         }
#         return batch

#     def __len__(self):
#         return len(self.full_list)

class SingleModalityDataset(Dataset):
    """
    Loops through triplets of (protein-sequence, text, reaction) from test datasets.
    Used for extraction of representations.
    """
    def __init__(self, path, tokenizer, max_sequence_len, modality = 'protein'):
        self.path = path
        self.modality = modality
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.df = pd.read_csv(path).reset_index()

        if modality == 'protein':
            self.sequence_list = self.df['sequence'].values.tolist()
            self.sequence_list = [" ".join(protein_sequence) for protein_sequence in self.sequence_list]
            #self.df['sequence'] = self.sequence_list
        elif modality == 'reaction':
            self.sequence_list = self.df['reaction_smiles'].values.tolist()
        elif modality == 'text':
            self.ec2text = pd.read_csv('../../data/PECT/full_datasets/EC2GOtext.csv').set_index('EC').to_dict()['desc']
            self.sequence_list = self.df['brenda'].map(self.ec2text).values.tolist()
        return

    def __getitem__(self, index):
        sequence = self.sequence_list[index]

        sequence_input_ids, sequence_attention_mask = encode_sequence(sequence, self.tokenizer, self.max_sequence_len)
        

        batch = {
            "sequence": sequence,
            "sequence_input_ids": sequence_input_ids,
            "sequence_attention_mask": sequence_attention_mask,
        }

        return batch
    
    def __len__(self):
        return len(self.sequence_list)
    
class CREEPDatasetMineBatch(Dataset):
    """
    Dataset is the length of the number of unique EC numbers.
    Use a batch size of 1 here and n_neg to determine the number of negative examples to include in the batch.
    Ensures that the rest of the batch is negative protein examples.
    Used for training
    """
    def __init__(self, path, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len, n_neg, loop_over_unique_EC=False, promiscuous_weight = 0.3):
        self.promiscuous_weight = promiscuous_weight
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len

        # self.df = pd.read_csv(path).reset_index()

        # self.protein_sequence_list = self.df['sequence'].values.tolist()
        # self.protein_sequence_list = [" ".join(protein_sequence) for protein_sequence in self.protein_sequence_list]
        # self.df['sequence'] = self.protein_sequence_list
        # self.text_sequence_list = self.df['name'].values.tolist()
        #self.reaction_sequence_list = self.df['reaction_smiles'].values.tolist()

        # print("num of protein-EC pairs: {}".format(len(self.protein_sequence_list)))
        
        # self.df['brenda'] = self.df['brenda'].str.split('.').str[:4].str.join('.') #number here is the number of levels of EC
        # #dictionary mapping from EC to smiles and vice versa
        # self.df['index'] = self.df.index
        #self.ec2index = self.df.groupby('brenda')['index'].apply(list).to_frame().to_dict()['index']
        self.ec2rxns = np.load(path + '/EC2rxns_train.npy', allow_pickle=True).item()
        self.ec2text = pd.read_csv('../../data/PECT/full_datasets/EC2GOtext.csv').set_index('EC').to_dict()['desc']
        self.ec2clusterid = np.load(path + '/EC2cluster70id_train.npy', allow_pickle=True).item()
        self.clusterid2proteinseq = np.load(path + '/cluster70id2proteinseq_train.npy', allow_pickle=True).item()
        self.clusterid2promiscuous = np.load(path + '/cluster70id2promiscuous_train.npy', allow_pickle=True).item()
        # self.clusterid2promiscuous_proteinseq = np.load(path + '/cluster70id2proteinseq_promiscuous_train.npy', allow_pickle=True).item()
        # self.clusterid2notpromiscuous_proteinseq = np.load(path + '/cluster70id2proteinseq_notpromiscuous_train.npy', allow_pickle=True).item()
        
        #add a space to the sequences in the dictionary (should potentially do this before if it's too slow)
        for key in self.clusterid2proteinseq.keys():
            self.clusterid2proteinseq[key] = [" ".join(protein_sequence) for protein_sequence in self.clusterid2proteinseq[key]]
        # for key in self.clusterid2promiscuous_proteinseq.keys():
        #     self.clusterid2promiscuous_proteinseq[key] = [" ".join(protein_sequence) for protein_sequence in self.clusterid2promiscuous_proteinseq[key]]
        # for key in self.clusterid2notpromiscuous_proteinseq.keys():
        #     self.clusterid2notpromiscuous_proteinseq[key] = [" ".join(protein_sequence) for protein_sequence in self.clusterid2notpromiscuous_proteinseq[key]]

        print("num of ECs: {}".format(len(self.ec2rxns)))
        print("num of reactions: {}".format(sum([len(self.ec2rxns[ec]) for ec in self.ec2rxns.keys()])))
        print("num of proteins: {}".format(sum([len(self.clusterid2proteinseq[key]) for key in self.clusterid2proteinseq.keys()])))

        # #print(len(ec_smiles['1.1']))
        # self.protein_sequence2ec = self.df.groupby('sequence')['brenda'].apply(list).to_frame().to_dict()['brenda']

        self.n_negs = n_neg
        self.batch_size = n_neg + 1
        self.loop_over_unique_ec = loop_over_unique_EC
        #loop by unique ECs
        if loop_over_unique_EC:
            self.full_list = []
            for ec in self.ec2rxns.keys(): #alternatively loop through all unique ECs instead of protein sequences
                if '-' not in ec:
                    self.full_list.append(ec)
        return
    
    def sample_protein(self, ec):
        """
        Uses clustering and proiscuous/non-promiscuous information to sample a protein sequence for increased diversity and promiscuity, for a given EC.
        """
        cluster_options = self.ec2clusterid[ec]
        clusterid = random.choice(cluster_options)

        choices = self.clusterid2proteinseq[clusterid]

        #TODO: random search weighted by promiscuity
        #probs = self.clusterid2promiscuous[clusterid]
        #np.random.choice(choices, p=probs)
        protein_sequence = random.choice(choices)

        # #promiscuous = random.choice([0,1]) #for now do 50/50
        # promiscuous = 1 if random.random() < self.promiscuous_weight else 0

        # if clusterid in self.clusterid2promiscuous_proteinseq.keys() and promiscuous == 1:
        #     protein_sequence = random.choice(self.clusterid2promiscuous_proteinseq[clusterid])
        # elif clusterid in self.clusterid2notpromiscuous_proteinseq.keys():
        #     protein_sequence = random.choice(self.clusterid2notpromiscuous_proteinseq[clusterid])
        # else:
        #     #last resort if the protein is always promiscuous
        #     protein_sequence = random.choice(self.clusterid2promiscuous_proteinseq[clusterid])

        return protein_sequence

    def __getitem__(self, index):

        if self.loop_over_unique_ec:
            anchor_ec = self.full_list[index]
            # anchor_protein_sequence = self.sample_protein(anchor_ec)
            #anchor_text = self.ec2text[anchor_ec]

        #choose randomly based on either choosing from (1) protein 
        #TODO: ensure that there are no overlaps in EC
        # negative_options = self.df[self.df['brenda'] != anchor_ec].index #this might be quite slow
        # negative_idxs = random.sample(list(negative_options), self.n_negs)

        #alternatively choose from (2) EC numbers without replacement
        other_ecs = list(self.ec2rxns.keys())
        other_ecs.remove(anchor_ec)
        negative_ecs = random.sample(other_ecs, self.n_negs) #sample EC numbers without replacement
        ecs = [anchor_ec]
        ecs.extend(negative_ecs)

        all_protein_sequence_input_ids = torch.zeros(self.batch_size, self.protein_max_sequence_len, dtype=torch.long)
        all_protein_sequence_attention_mask = torch.zeros(self.batch_size, self.protein_max_sequence_len, dtype=torch.long)
        all_text_sequence_input_ids = torch.zeros(self.batch_size, self.text_max_sequence_len, dtype=torch.long)
        all_text_sequence_attention_mask = torch.zeros(self.batch_size, self.text_max_sequence_len, dtype=torch.long)
        all_reaction_sequence_input_ids = torch.zeros(self.batch_size, self.reaction_max_sequence_len, dtype=torch.long)
        all_reaction_sequence_attention_mask = torch.zeros(self.batch_size, self.reaction_max_sequence_len, dtype=torch.long)

        for i,  ec in enumerate(ecs):
            protein_sequence = self.sample_protein(ec)
            text_sequence = self.ec2text[ec]
            reaction_sequence = random.choice(self.ec2rxns[ec])

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