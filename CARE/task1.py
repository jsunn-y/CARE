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


def make_fasta(df: pd.DataFrame, filepath: str, seq_column='Sequence', id_column='Entry'):
    """
    Make a fasta file for clustering the data for mmSeqs.
    """
    # generate a fasta file as input to mmseqs
    with open(filepath, 'w') as f:
        for index, row in df.iterrows():
            f.write(f'>{row[id_column]}\n{row[seq_column]}\n')
    
def cluster_mmseqs(input_fasta_filename: str, output_fasta_filename: str, cluster_resolution: float, tmp_dir='/tmp'): 
    """
    Cluster using the mmSeqs2 algorithm. https://github.com/soedinglab/MMseqs2
    """
    # ToDo: add in warnings for uninstalled or for bad arguments.
    os.system(f'mmseqs easy-cluster {input_fasta_filename} {output_fasta_filename} {tmp_dir} --min-seq-id {cluster_resolution}')

def process_mmseqs_clustering(cluster_fasta_filename: str, cluster_resolution: float, name=None):
    """
    Load the data from mmseqs and make it into a dataframe.
    """
    name = name if name else f'clusterRes{cluster_resolution}'
    clustering = pd.read_csv(cluster_fasta_filename, delimiter='\t', header=None)
    #rename heading as cluster reference and id
    clustering.columns = [name, 'Entry']
    clustering.drop_duplicates(subset='Entry', keep='first', inplace=True)
    print(clustering[name].nunique())
    clustering[name].value_counts().hist(bins=100)
    return clustering


    

def process_uniprot(uniprot_path: str, reaction2EC: pd.DataFrame, min_length=100, max_length=650):
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
    # Filter to only include the seqs that have an EC number overlapping with a reaction
    swissprot = swissprot[swissprot['EC number'].isin(reaction2EC['EC number'].unique())]
    return swissprot

