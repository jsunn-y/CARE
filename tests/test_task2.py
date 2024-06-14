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
from CARE.task2 import *

repo_dir = '/disk1/ariane/vscode/CARE/pretrained/'
data_dir = f'{repo_dir}raw_data/'
test_data_dir = f'{repo_dir}tests/data/'
output_dir = f'/disk1/ariane/vscode/CARE/tests/output/'
task2_dir = f'{repo_dir}splits/task2/'
processed_dir = f'{repo_dir}processed_data/'

class TestTask2(unittest.TestCase):

    def test_random(self):
        tasker = Task2(task2_dir, output_dir, f'{processed_dir}text2EC.csv')
        tasker.randomly_assign_EC('easy', num_ecs=50, save=True, run_tag='random_')

    def test_chatGPT(self):
        tasker = Task2(output_dir, output_dir)
        
        with open(f'{repo_dir}secrets.txt', 'r+') as fin:
            for line in fin:
                api_key = line.strip()
                break
        df = tasker.get_ChatGPT('promiscuous', api_key=api_key, subsample=['Q48154', 'A0JNI4', 'Q62969'], save=True)
    
    def test_similarity(self):
        tasker = Task2(task2_dir, output_dir, processed_dir, pretrained_dir=f'{repo_dir}task2_baselines/')
        df = tasker.get_test_df('easy').sample(2)
        # Pass the DF just so that we can easily do this quickly
        # reference_dataset, query_dataset, pretrained_folder, output_folder, reference_modality, query_modality,
        tasker.get_similarity('Similarity', 'easy', encode=True, df=df)
        #print(tasker.downstream_retrieval('all_ECs', 'easy', 'reaction', 'reaction'))
        print(tasker.tabulate_results('Similarity', 'easy'))

    def test_CREEP(self):
        tasker = Task2(task2_dir, output_dir, processed_dir, pretrained_dir=f'{repo_dir}task2_baselines/')

        print(tasker.downstream_retrieval('CREEP', 'all_ECs', 'easy', 'reaction', 'reaction'))
        print(tasker.tabulate_results('CREEP', 'easy'))

    def test_ClipZyme(self):
        tasker = Task2(task2_dir, output_dir, processed_dir, pretrained_dir=f'{repo_dir}task2_baselines/')
        print(tasker.downstream_retrieval('CLIPZyme', 'all_ECs', 'medium', 'protein', 'reaction'))
        print(tasker.tabulate_results('CLIPZyme', 'medium'))


if __name__ == '__main__':
    unittest.main()