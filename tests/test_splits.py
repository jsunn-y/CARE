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

import sys
sys.path.append('/disk1/ariane/vscode/CARE/CARE/')
sys.path.append('/disk1/ariane/vscode/CARE/')

from CARE.processing import *
from CARE.task1 import *

data_dir = '/disk1/ariane/vscode/CARE/data/raw/'
test_data_dir = '/disk1/ariane/vscode/CARE/tests/data/'
output_dir = '/disk1/ariane/vscode/CARE/tests/output/'

class TestSplits(unittest.TestCase):

    def test_split(self):
        split(pd.read_csv(f'{output_dir}protein2EC.csv'), f'{data_dir}price.tsv', output_dir)

    def test_random(self):
        tasker = Task1(output_dir, output_dir)
        tasker.randomly_assign_EC('promiscuous', num_ecs=10, save=True, run_tag='random_')

    def test_blast(self):
        tasker = Task1(output_dir, output_dir)
        tasker.get_blast('promiscuous', 10, save=True, run_tag='BLAST_')
        df = pd.read_csv(f'{output_dir}BLAST_promiscuous_protein_test_results_df.csv')
        # Check some of these manually. 
        assert '2.7.7.40' in ' '.join(df[df['Entry'] == 'Q48154']['EC number'].values)
        print(df[df['Entry'] == 'Q48154']['EC number'].values)
        assert '1.1.1.405' in ' '.join(df[df['Entry'] == 'Q48154']['EC number'].values)
        # 5.3.1.1,4.2.3.3
        assert '4.2.3.3' in ' '.join(df[df['Entry'] == 'P17751']['EC number'].values)
        assert '5.3.1.1' in ' '.join(df[df['Entry'] == 'P17751']['EC number'].values)

    def test_blast(self):
        # Check that our getting of the correct ECs is right at least, and then it is just the algorithm for anything else
        tasker = Task1(output_dir, output_dir)
        tasker.get_blast('30-50', 1, save=True, run_tag='BLAST_')
        df = pd.read_csv(f'{output_dir}BLAST_30-50_protein_test_results_df.csv')

        # Check 3 for each
        # G3F5K2,BKT_PROBT,387,,1.14.99.63,1.14.99.63,G3F5K2,G3F5K2,G3F5K2,G3F5K2,1.14.99,1.14,1
        # Q93Z04,PLY13_ARATH,501,,4.2.2.2,4.2.2.2,Q93Z04,Q93Z04,Q93Z04,Q93Z04,4.2.2,4.2,4
        # Q95XM2,PHM_CAEEL,324,,1.14.17.3,1.14.17.3,Q95XM2,Q95XM2,Q95XM2,Q95XM2,1.14.17,1.14,1
        assert '1.14.99.63'  == df[df['Entry'] == 'G3F5K2']['EC number'].values[0]
        assert '4.2.2.2'  == df[df['Entry'] == 'Q93Z04']['EC number'].values[0]
        assert '1.14.17.3'  == df[df['Entry'] == 'Q95XM2']['EC number'].values[0]

        # Do the same for promisc 
        tasker = Task1(output_dir, output_dir)
        tasker.get_blast('promiscuous', 1, save=True, run_tag='BLAST_')
        df = pd.read_csv(f'{output_dir}BLAST_promiscuous_protein_test_results_df.csv')
        #c5545,Q48154,,1.1.1.405;2.7.7.40,4,2,True,False
        # 94,A0JNI4,,4.3.1.17;4.3.1.18;5.1.1.18,4,3,True,False
        # 6254,Q62969,,4.2.1.152;5.3.99.4,4,2,True,False
        assert '1.1.1.405;2.7.7.40'  == df[df['Entry'] == 'Q48154']['EC number'].values[0]
        assert '4.3.1.17;4.3.1.18;5.1.1.18'  == df[df['Entry'] == 'A0JNI4']['EC number'].values[0]
        assert '4.2.1.152;5.3.99.4'  == df[df['Entry'] == 'Q62969']['EC number'].values[0]
        
        tasker = Task1(output_dir, output_dir)
        tasker.get_blast('price', 1, save=True, run_tag='BLAST_')
        df = pd.read_csv(f'{output_dir}BLAST_price_protein_test_results_df.csv')
        # Do the same for Price
        # WP_012434865,4.2.1.25,,577
        # WP_011717064,3.1.1.15,,300
        # WP_010207016,1.3.8.7,,601
        assert '4.2.1.25'  == df[df['Entry'] == 'WP_012434865']['EC number'].values[0]
        assert '3.1.1.15'  == df[df['Entry'] == 'WP_011717064']['EC number'].values[0]
        assert '1.3.8.7'  == df[df['Entry'] == 'WP_010207016']['EC number'].values[0]
        

    def test_blast_easy_match(self):

        tasker = Task1(output_dir, output_dir)
        tasker.get_blast('70-90', 10, save=True, run_tag='BLAST_')
        df = pd.read_csv(f'{output_dir}BLAST_70-90_protein_test_results_df.csv')
        # Check that the predicted ones are in the same cluster Res 90
        entry = 'Q9UAG7'
        predicted = df[df['Entry'] == entry]['Similar Enzymes'].values[0].split(';')
        print(predicted)
        df = pd.read_csv(f'{output_dir}protein2EC.csv')
        protein2cluster = dict(zip(df['Entry'], df['clusterRes30']))
        # Check they have the same clusterRes30 group at least
        for p in predicted:
            print(protein2cluster.get(p), protein2cluster.get(entry))
            assert protein2cluster.get(p) == protein2cluster.get(entry)

        # Also check the EC is correctly predicted
        entry = 'P41213'
        predicted = df[df['Entry'] == entry]['EC number'].values[0].split(';')
        print(predicted)
        df = pd.read_csv(f'{output_dir}protein2EC.csv')
        protein2ec = dict(zip(df['Entry'], df['EC number']))
        # Check they have the same clusterRes30 group at least
        for p in predicted:
            print(p, protein2ec.get(entry))
            assert p in protein2ec.get(entry)


    def test_proteInfer(self):
        tasker = Task1(output_dir, output_dir)
        tasker.get_proteinfer('promiscuous', proteinfer_dir='/disk1/ariane/pycharm/CARE/proteinfer/', run_tag='ProteInfer_', save=True)
        df = pd.read_csv(f'{output_dir}ProteInfer_promiscuous_protein_test_results_df.csv')
        # Check some of these manually. 
        assert '2.7.7.40' in ' '.join(df[df['Entry'] == 'Q48154']['EC number'].values)
        print(df[df['Entry'] == 'Q48154']['EC number'].values)
        assert '1.1.1.405' in ' '.join(df[df['Entry'] == 'Q48154']['EC number'].values)
        # 5.3.1.1,4.2.3.3
        assert '4.2.3.3' in ' '.join(df[df['Entry'] == 'P17751']['EC number'].values)
        assert '5.3.1.1' in ' '.join(df[df['Entry'] == 'P17751']['EC number'].values)
        # Check when we have > 90% similarity that we are able to get some of these ones very accurately.
        tasker.get_proteinfer('70-90', proteinfer_dir='/disk1/ariane/pycharm/CARE/proteinfer/', save=True, run_tag='ProteInfer_')
        df = pd.read_csv(f'{output_dir}ProteInfer_70-90_protein_test_results_df.csv')
        # Check that the predicted ones are in the same cluster Res 90
        entry = 'Q9UAG7'
        predicted = df[df['Entry'] == entry]['EC number'].values[0].split(';')
        print(predicted)
        df = pd.read_csv(f'{output_dir}protein2EC.csv')
        protein2ec = dict(zip(df['Entry'], df['EC number']))
        # Check they have the same clusterRes30 group at least
        for p in predicted:
            print(p, protein2ec.get(entry))
            assert p in protein2ec.get(entry)


    def test_chatGPT(self):
        tasker = Task1(output_dir, output_dir)
        
        with open('/disk1/ariane/vscode/CARE/secrets.txt', 'r+') as fin:
            for line in fin:
                api_key = line.strip()
                break
        df = tasker.get_ChatGPT('promiscuous', api_key=api_key, subsample=['Q48154', 'A0JNI4', 'Q62969'], save=True)
         #c5545,Q48154,,1.1.1.405;2.7.7.40,4,2,True,False
        # 94,A0JNI4,,4.3.1.17;4.3.1.18;5.1.1.18,4,3,True,False
        # 6254,Q62969,,4.2.1.152;5.3.99.4,4,2,True,False
        assert '1.1.1.405;2.7.7.40'  == df[df['Entry'] == 'Q48154']['EC number'].values[0]
        assert '4.3.1.17;4.3.1.18;5.1.1.18'  == df[df['Entry'] == 'A0JNI4']['EC number'].values[0]
        assert '4.2.1.152;5.3.99.4'  == df[df['Entry'] == 'Q62969']['EC number'].values[0]
    
if __name__ == '__main__':
    unittest.main()