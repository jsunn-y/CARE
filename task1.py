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
import sys
import os
sys.path.append(f'{str(os.getcwd())}/CARE/')
sys.path.append(os.getcwd())

from CARE.processing import *
from CARE.task1 import *
from CARE.task2 import *
from CARE.benchmarker import *

from sciutil import *

u = SciUtil()
import argparse

care_dir = '/disk1/ariane/vscode/CARE/pretrained/'
processed_data_dir = f'{care_dir}processed_data/'
raw_data_dir = f'{care_dir}raw_data/'
output_dir = f'{care_dir}user_output/'
task1_data_dir = f'{care_dir}splits/task1/'
task2_data_dir = f'{care_dir}splits/task2/'


def run_tasks(splits, run_blast, run_proteinfer, run_random, run_chatGPT, run_process, run_split, output_dir):

    # Process the data
    if run_process:
        process_data(f'{processed_data_dir}uniprotkb_AND_reviewed_true_2024_05_13.tsv', f'{processed_data_dir}ECReact.csv', 
                    f'{processed_data_dir}EnzymeMap.csv', f'{processed_data_dir}ECtoGO_raw.txt', output_dir)

    # Make splits for task1
    if run_split:
        split(pd.read_csv(f'{output_dir}protein2EC.csv'), f'{data_dir}price.tsv', task1_data_dir)

    # Now let's run each tool and then benchmark
    # Save in the required format
    # Make a task1
    # Build the df of the rows for this

    rows = []
    if run_blast:
        output_dir = f'{care_dir}task1_baselines/results_summary/BLAST/'
        tasker = Task1(data_folder=task1_data_dir, output_folder=output_dir, processed_data_folder=processed_data_dir)
        for split in splits:
            # For proteInfer you need to point where it was saved.
            df = tasker.get_blast(split, num_ecs=10, save=True)
            u.dp([split])
            rows = get_k_acc(df, [1, 5, 10], rows, 'BLAST', split)

    # Then check the results for random as well
    if run_random:
        output_dir = f'{care_dir}results_summary/Random/'
        tasker = Task1(data_folder=task1_data_dir, output_folder=output_dir)
        for split in splits:
            # For proteInfer you need to point where it was saved.
            df = tasker.randomly_assign_EC(split, num_ecs=50, save=True)
            u.dp([split])
            rows = get_k_acc(df, [1, 5, 10], rows, 'random', split)

        # Also do it for the random for task 2
        output_dir = f'{care_dir}task2_baselines/results_summary/Random/'
        tasker = Task2(data_folder=task2_data_dir, output_folder=output_dir, ec2text=f'{processed_data_dir}text2EC.csv')
        for split in ['easy', 'medium', 'hard']:
            # For proteInfer you need to point where it was saved.
            df = tasker.randomly_assign_EC(split, num_ecs=50, save=True)
            u.dp([split])
            rows = get_k_acc(df, [1, 5, 10], rows, 'random', split)

    # Then check the results for random as well
    if run_proteinfer:
        output_dir = f'{care_dir}task1_baselines/results_summary/ProteInfer/'
        tasker = Task1(data_folder=task1_data_dir, output_folder=output_dir)
        for split in splits:
            # For proteInfer you need to point where it was saved.
            df = tasker.get_proteinfer(split, proteinfer_dir=f'{task1_data_dir}task1_baselines/ProteInfer/proteinfer/', save=True)
            u.dp([split])
            rows = get_k_acc(df, [1, 5, 10], rows, 'ProteInfer', split)


    if run_chatGPT:      
        output_dir = f'{care_dir}task1_baselines/results_summary/ChatGPT/'
        tasker = Task1(task1_data_dir, output_dir)
        with open(f'{care_dir}secrets.txt', 'r+') as fin:
            for line in fin:
                api_key = line.strip()
                break
        for split in splits:
            df = tasker.get_ChatGPT(split, api_key=api_key, save=True)
            u.dp([split])
            rows = get_k_acc(df, [1, 5, 10], rows, 'ChatGPT', split)

    df = pd.DataFrame(rows, columns=['baseline', 'split', 'k', 'level 4 accuracy', 'level 3 accuracy', 'level 2 accuracy', 'level 1 accuracy'])
    df.to_csv(f'{task1_data_dir}results_all.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", type=str, default=os.getcwd())
    parser.add_argument("--split", type=str, choices=['30', '30-50', 'price', 'promiscuous'])
    parser.add_argument("--tool", type=str, default='BLAST', choices=['BLAST', 'ProteInfer', 'ChatGPT', 'Random'])

    args = parser.parse_args()

    # def run_tasks(splits, run_blast, run_proteinfer, run_random, run_chatGPT, output_dir):

    if args.tool == 'BLAST':
        print(args.split)
        args.split = '30'
        run_tasks([args.split], True, False, False, False, False, False, output_dir)
