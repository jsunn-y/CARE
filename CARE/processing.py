#################################################################################
# Copyright (c) 2012-2024 Scott Chacon and others                               #
#                                                                               #    
# Permission is hereby granted, free of charge, to any person obtaining         #
# a copy of this software and associated documentation files (the               #
# "Software"), to deal in the Software without restriction, including           #
# without limitation the rights to use, copy, modify, merge, publish,           #
# distribute, sublicense, and/or sell copies of the Software, and to            #
# permit persons to whom the Software is furnished to do so, subject to         #
# the following conditions:                                                     #                                          
# The above copyright notice and this permission notice shall be                #
# included in all copies or substantial portions of the Software.               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,               #
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF            #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                         #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE        #  
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION        #
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION         #
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.               #
#################################################################################


import pandas as pd
import os


def process_data(uniprot_path: str, ec_react_path: str, enzymemap_path: str, ec2go_path: str, output_dir: str, min_length=100, max_length=650):
    """ 
    This is the main processing data function that processes everything from the raw inputs to the ones used in splitting.
    """
    protein_df = process_uniprot(uniprot_path)
    reaction_df = process_reactions(ec_react_path, enzymemap_path)
    reaction_df, protein_df = filter_for_overlapping_EC(protein_df, reaction_df)

    # Make fasta file, and save to the output dir
    make_fasta(protein_df, f'{output_dir}protein.fasta')

    cluster_mmseqs(f'{output_dir}protein.fasta', f'{output_dir}clusterRes30', cluster_resolution=0.3)
    cluster_mmseqs(f'{output_dir}protein.fasta', f'{output_dir}clusterRes50', cluster_resolution=0.5)
    cluster_mmseqs(f'{output_dir}protein.fasta', f'{output_dir}clusterRes70', cluster_resolution=0.7)
    cluster_mmseqs(f'{output_dir}protein.fasta', f'{output_dir}clusterRes90', cluster_resolution=0.9)

    df_30 = process_mmseqs_clustering(f'{output_dir}clusterRes30_cluster.tsv', name=f'clusterRes30')
    df_50 = process_mmseqs_clustering(f'{output_dir}clusterRes50_cluster.tsv', name=f'clusterRes50')
    df_70 = process_mmseqs_clustering(f'{output_dir}clusterRes70_cluster.tsv', name=f'clusterRes70')
    df_90 = process_mmseqs_clustering(f'{output_dir}clusterRes90_cluster.tsv', name=f'clusterRes90')

    protein_df = pd.merge(protein_df, df_30, on='Entry', how='left')
    protein_df = pd.merge(protein_df, df_50, on='Entry', how='left')
    protein_df = pd.merge(protein_df, df_70, on='Entry', how='left')
    protein_df = pd.merge(protein_df, df_90, on='Entry', how='left')

    protein_df['EC3'] = protein_df['EC number'].str.split('.').str[:3].str.join('.')
    protein_df['EC2'] = protein_df['EC number'].str.split('.').str[:2].str.join('.')
    protein_df['EC1'] = protein_df['EC number'].str.split('.').str[:1].str.join('.')

    # Save to a csv and then also do the go processing
    protein_df.to_csv(f'{output_dir}protein2EC.csv', index=False)

    #save ECs to a txt as the order of the cluster centers for downstream tasks
    ec2go = process_ec2go(ec2go_path, protein_df)

    with open(f'{output_dir}EC_list.txt', 'w') as f:
        for ec in ec2go['EC number']:
            f.write(ec + '\n')
    # yay done.

def get_EC_desc(EC, EC2desc):
    num_dashes = EC.count('-')

    description_missing = False
    EC1 = '.'.join(EC.split('.')[:1])
    if EC1 + '.-.-.-' in EC2desc and num_dashes < 3:
        desc1 = EC2desc[EC1 + '.-.-.-']
    else:
        desc1 = ''
    EC2 = '.'.join(EC.split('.')[:2]) 
    if EC2 + '.-.-' in EC2desc and num_dashes < 2:
        desc2 = EC2desc[EC2 + '.-.-']
    else:
        desc2 = ''
    EC3 = '.'.join(EC.split('.')[:3])
    if EC3 + '.-' in EC2desc and num_dashes < 1:
        desc3 = EC2desc[EC3 + '.-']
    else:
        desc3 = ''
        
    if EC in EC2desc:
        desc4 = EC2desc[EC]
    else:
        description_missing = True
        desc4 = ''

    description = desc1 + '; ' + desc2 + '; ' + desc3 + '; ' + desc4
    description = description.replace(' ; ', ' ')
    description = description.replace(' ; ', ' ')
    description = description.replace(' ; ', ' ')
    description = description.replace(' activity', '')
    #if string starts with ;, replace with space
    if description[0] == ';':
        description = description[2:]
    if description[-2:] == '; ':
        description = description[:-2]
    return description

def process_ec2go(ec2go_path: str, protein_df: pd.DataFrame, ec_column='EC number'):
    """
    Process GO terms.
    raw_data/ECtoGO_raw.txt was downloaded from http://current.geneontology.org/ontology/external2go/ec2go
    """
    #read ECtoGO.txt line by line
    ECtoGO = open(ec2go_path, "r")
    ECtoGO_lines = ECtoGO.readlines()
    ECtoGO.close()

    #skip the first two lines
    ECtoGO_lines = ECtoGO_lines[2:]
    EC2desc = {}
    for line in ECtoGO_lines:
        line = line.strip().split(">")
        EC = line[0][3:-1]
        desc = line[1].split(";")[0][4:-1]
        EC2desc[EC] = desc

    #subset to the EC numbers in swissprot
    ec2go_df = pd.DataFrame(protein_df[ec_column].unique(), columns=[ec_column])
    ec2go_df = ec2go_df.sort_values(by=ec_column)
    ec2go_df = ec2go_df[ec2go_df[ec_column].isin(protein_df[ec_column].unique())]
    ec2go_df['Text'] = [get_EC_desc(ec, EC2desc) for ec in ec2go_df[ec_column].values]
    ec2go_df['Text Incomplete'] = ~ec2go_df[ec_column].isin(ec2go_df.keys())

    return ec2go_df


def process_reactions(ec_react_path: str, enzymemap_path: str):
    ec_react = pd.read_csv(ec_react_path)
    ec_react['rxn_smiles'] = ec_react['rxn_smiles'].str.split('|').str[0] + '>>' + ec_react['rxn_smiles'].str.split('>>').str[1]
    ec_react.rename(columns={'rxn_smiles': 'Reaction', 'ec': 'EC number', 'source':'Source'}, inplace=True)
    
    enzymemap = pd.read_csv(enzymemap_path)
    enzymemap.rename(columns={'ec_num': 'EC number', 'unmapped': 'Reaction', 'mapped': 'Mapped Reaction', 'orig_rxn_text':'Reaction Text'}, inplace=True)
    enzymemap = enzymemap[['Reaction', 'Mapped Reaction', 'EC number', 'Reaction Text']]
    enzymemap.drop_duplicates(subset=['Reaction', 'EC number'], inplace=True)
    enzymemap = enzymemap.dropna(subset=['Mapped Reaction']) #just double check that all of them have mapped reactions

    #find the EC numbers covered by ECreact but not by enzymemap
    ec_react = ec_react[~ec_react['EC number'].str.contains('-')]
    not_covered = ec_react[~ec_react['EC number'].isin(enzymemap['EC number'].unique())]
    reaction2EC = pd.concat((enzymemap, not_covered[['Reaction', 'EC number']])).sort_values(by='EC number')
    
    return reaction2EC

def filter_for_overlapping_EC(reaction_df: pd.DataFrame, protein_df: pd.DataFrame, ec_column='EC number'):
    """
    Filter to only include ECs that have both a reaction and a sequence entry. 
    """
    protein_df = protein_df[protein_df[ec_column].isin(reaction_df[ec_column].unique())]
    reaction_df = reaction_df[reaction_df[ec_column].isin(protein_df[ec_column].unique())]

    return protein_df, reaction_df

def make_fasta(df: pd.DataFrame, filepath: str, seq_column='Sequence', id_column='Entry'):
    """
    Make a fasta file for clustering the data for mmSeqs.
    """
    # generate a fasta file as input to mmseqs
    with open(filepath, 'w') as f:
        for index, row in df.iterrows():
            f.write(f'>{row[id_column]}\n{row[seq_column]}\n')
    
def cluster_mmseqs(input_fasta_filename: str, name: str, cluster_resolution: float, tmp_dir='/tmp'): 
    """
    Cluster using the mmSeqs2 algorithm. https://github.com/soedinglab/MMseqs2
    """
    # ToDo: add in warnings for uninstalled or for bad arguments.
    print(f'mmseqs easy-cluster {input_fasta_filename} {name} {tmp_dir} --min-seq-id {cluster_resolution}')
    os.system(f'mmseqs easy-cluster {input_fasta_filename} {name} {tmp_dir} --min-seq-id {cluster_resolution}')

def process_mmseqs_clustering(cluster_fasta_filename: str, name: str):
    """
    Load the data from mmseqs and make it into a dataframe.
    """
    clustering = pd.read_csv(cluster_fasta_filename, delimiter='\t', header=None)
    #rename heading as cluster reference and id
    clustering.columns = [name, 'Entry']
    clustering.drop_duplicates(subset='Entry', keep='first', inplace=True)
    clustering[name].value_counts().hist(bins=100)
    return clustering

def process_uniprot(uniprot_path: str, min_length=100, max_length=650):
    """
    Process uniprot data that was downlaed by the following means: 

    1. Downloaded the data on 13th of May 2024. 571,282 results filtering for reviewed "Swiss-Prot". (https://www.uniprot.org/uniprotkb?query=*&facets=reviewed%3Atrue)
    2. Selected download TSV and the columns, Seqeunce (under Sequences tab), EC number (under Function). Unzipped the downloaded file.

    Filtering steps:
    1. Filter to include only proteins with an EC number annotated
    2. Filter to only include proteins with a certain length
    3. Only include ECs with non - characters (i.e. complete ECs)
    4. Only include ECs that have an annotated reaction.

    uniprot_path: path to uniprot file
    min_length: min allowable protein length (aa)
    max_length: max allowable protein length
    """
    swissprot = pd.read_csv(uniprot_path, sep='\t')
    # explode out data
    swissprot = swissprot[swissprot['EC number'].notna()]
    # Drop rows that don't have an ec number
    swissprot['EC All'] = swissprot['EC number'].values
    # Now expand out the ones we have left
    swissprot['EC number'] = [ec.split(';') for ec in swissprot['EC number']]
    # Then check how many have mulitple
    swissprot = swissprot.explode('EC number')
    # Clean the EC numbers
    swissprot['EC number'] = [ec.replace(' ', '') for ec in swissprot['EC number']]
    swissprot = swissprot[swissprot['Length'] > min_length]
    swissprot = swissprot[swissprot['Length'] < max_length]
    swissprot = swissprot[~swissprot['EC number'].str.contains('-')]
    swissprot = swissprot.drop_duplicates(subset=['Sequence', 'EC number'])
    return swissprot

