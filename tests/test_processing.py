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

import unittest

from CARE.processing import *
from CARE.task1 import *
from CARE.task2 import *
from CARE.benchmarker import *

import sys
sys.path.append('/disk1/ariane/vscode/CARE/CARE/')

from CARE.processing import *

data_dir = '/disk1/ariane/vscode/CARE/data/'
test_data_dir = '/disk1/ariane/vscode/CARE/tests/data/'

care_dir = '/disk1/ariane/vscode/CARE/pretrained/'
processed_data_dir = f'{care_dir}processed_data/'
raw_data_dir = f'{care_dir}raw_data/'
output_dir = f'{care_dir}user_output/'
task1_data_dir = f'{care_dir}splits/task1/'
task2_data_dir = f'{care_dir}splits/task2/'

class TestProcessing(unittest.TestCase):

    def test_uniprot(self):
        sp = process_uniprot(f'{data_dir}uniprotkb_AND_reviewed_true_2024_05_13.tsv')
        # Check the length
        assert len(sp) == 183162
        sp.to_csv(f'{test_data_dir}sp_df.csv')

    def test_reaction(self):
        ecdf = process_reactions(f'{data_dir}ECReact.csv', f'{data_dir}EnzymeMap.csv')
        print("DONE")
        print(ecdf.head())
        assert len(ecdf) == 72490
        ecdf.to_csv(f'{test_data_dir}ec_df.csv')

    def test_ec_filter(self):
        ec_df, sp_df = filter_for_overlapping_EC(pd.read_csv(f'{test_data_dir}sp_df.csv'), pd.read_csv(f'{test_data_dir}ec_df.csv'))
        print(len(ec_df), len(sp_df))
        assert len(ec_df) == 59122
        assert len(sp_df) == 169865
        ec_df.to_csv(f'{test_data_dir}reaction_df.csv')
        sp_df.to_csv(f'{test_data_dir}protein_df.csv')

    def test_cluster(self):
        make_fasta(pd.read_csv(f'{test_data_dir}protein_df.csv'), f'{test_data_dir}protein.fasta')

    def test_mmseqs(self):
        # Adds .fasta onto it
        cluster_mmseqs(f'{test_data_dir}protein.fasta', f'{test_data_dir}clusterRes30', cluster_resolution=0.3)

    def test_cluster_pprocessing(self):
        # Read in from the previous
        clusters = process_mmseqs_clustering(f'{test_data_dir}clusterRes30_cluster.tsv', name=f'clusterRes30')
        df = pd.read_csv(f'{test_data_dir}protein_df.csv')
        # Check we can join and the number of clusters is correct 
        df = pd.merge(df, clusters, on='Entry', how='left')
        # Check the numbers are correct
        assert clusters['clusterRes30'].nunique() == 9012
        df.to_csv(f'{test_data_dir}protein_df_clustered.csv')

    def test_ecgo(self):
        ec2go = process_ec2go(f'{data_dir}ECtoGO_raw.txt', pd.read_csv(f'{test_data_dir}protein_df.csv'))
        assert len(ec2go) == 4673

    def test_pipeline(self):
        process_data(f'{data_dir}uniprotkb_AND_reviewed_true_2024_05_13.tsv', f'{data_dir}ECReact.csv', f'{data_dir}EnzymeMap.csv', f'{data_dir}ECtoGO_raw.txt', '/disk1/ariane/vscode/CARE/tests/output/')

    def test_blast_pipeline(self):
        output_dir = f'{care_dir}task1_baselines/results_summary/BLAST/'
        tasker = Task1(data_folder=task1_data_dir, output_folder=output_dir)
        split = '30-50'
        # For proteInfer you need to point where it was saved.
        df = tasker.get_blast(split, num_ecs=10, save=True)
        u.dp([split])


if __name__ == '__main__':
    unittest.main()
