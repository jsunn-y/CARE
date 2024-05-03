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

    ###TODO: mine hard negative examples
    neg_protein_sequence = random.choice(ec2protein_sequence[neg_ec])
    return neg_protein_sequence

class SingleModalityDataset(Dataset):
    """
    Loops through triplets of (protein-sequence, text, reaction) from test datasets.
    Used for extraction of representations.
    """
    def __init__(self, file, tokenizer, max_sequence_len, modality = 'protein'):
        self.file = file
        self.modality = modality
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.df = pd.read_csv(file).reset_index()

        if modality == 'protein':
            self.sequence_list = self.df['Sequence'].values.tolist()
            self.sequence_list = [" ".join(protein_sequence) for protein_sequence in self.sequence_list]
            #self.df['sequence'] = self.sequence_list
        elif modality == 'reaction':
            self.sequence_list = self.df['Reaction'].values.tolist()
        elif modality == 'text':
            text2EC_df = pd.read_csv('../../processed_data/text2EC.csv')
            self.ec2text = text2EC_df.set_index('EC number').to_dict()['Text']
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
    def __init__(self, dataset_path, split_file, protein_tokenizer, text_tokenizer, reaction_tokenizer, protein_max_sequence_len, text_max_sequence_len, reaction_max_sequence_len, n_neg):
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len

        protein2EC_df = pd.read_csv(dataset_path + 'protein2EC.csv')
        reaction2EC_df = pd.read_csv(dataset_path + 'reaction2EC.csv')
        text2EC_df = pd.read_csv(dataset_path + 'text2EC.csv')
        
        #load the train indices from a txt and subsample the reactions
        train_indices = np.loadtxt(split_file, dtype=int)
        reaction2EC_df = reaction2EC_df.iloc[train_indices]

        self.ec2text = text2EC_df.set_index('EC number').to_dict()['Text'] #one to one mapping
        self.ec2rxns = reaction2EC_df.groupby('EC number')['Reaction'].apply(list).to_frame().to_dict()['Reaction']

        #for now use the 50% identity clusters
        self.ec2clusterid = protein2EC_df.groupby('EC number')['clusterRes50'].apply(list).to_frame().to_dict()['clusterRes50']
        self.clusterid2proteinseq = protein2EC_df.groupby('clusterRes50')['Sequence'].apply(list).to_frame().to_dict()['Sequence']
        
        #add a space to the sequences in the dictionary (should potentially do this before if it's too slow)
        for key in self.clusterid2proteinseq.keys():
            self.clusterid2proteinseq[key] = [" ".join(protein_sequence) for protein_sequence in self.clusterid2proteinseq[key]]
        
        print("num of ECs: {}".format(len(self.ec2rxns)))
        print("num of reactions: {}".format(sum([len(self.ec2rxns[ec]) for ec in self.ec2rxns.keys()])))
        print("num of proteins: {}".format(sum([len(self.clusterid2proteinseq[key]) for key in self.clusterid2proteinseq.keys()])))

        self.n_negs = n_neg
        self.batch_size = n_neg + 1
        # self.loop_over_unique_ec = loop_over_unique_EC
        # #loop by unique ECs
        # if loop_over_unique_EC:
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

        # if self.loop_over_unique_ec:
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