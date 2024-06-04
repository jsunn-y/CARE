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
sys.path.append('/disk1/ariane/vscode/CARE/CARE/')
sys.path.append('/disk1/ariane/vscode/CARE/')

from CARE.processing import *
from CARE.task1 import *

data_dir = '/disk1/ariane/vscode/CARE/data/raw/'
output_dir = '/disk1/ariane/vscode/CARE/data/processed/'
task1_dir = '/disk1/ariane/vscode/CARE/data/task1/'
task2_dir = '/disk1/ariane/vscode/CARE/data/task2/'

# Process the data
process_data(f'{data_dir}uniprotkb_AND_reviewed_true_2024_05_13.tsv', f'{data_dir}ECReact.csv', 
             f'{data_dir}EnzymeMap.csv', f'{data_dir}ECtoGO_raw.txt', output_dir)

# Make splits for task1
split(pd.read_csv(f'{output_dir}protein2EC.csv'), f'{data_dir}price.tsv', task1_dir)
