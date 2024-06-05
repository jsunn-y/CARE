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

from CARE.processing import *
from CARE.task1 import *
from CARE.benchmarker import *

data_dir = '/disk1/ariane/vscode/CARE/data/'
test_data_dir = '/disk1/ariane/vscode/CARE/tests/data/'
output_dir = '/disk1/ariane/vscode/CARE/tests/output/'

class TestBenchmark(unittest.TestCase):

    def test_acc(self):
        print(get_accuracy_level(["1.1.1.1"], ["1.1.1.1"]))
        values = get_accuracy_level(["1.1.1.1"], ["1.1.1.1"])[0]
        assert sum(values) == 4
        test2 = get_accuracy_level(["1.1.1.0"], ["1.1.1.1"])[0]
        assert sum(test2) == 3
        test3 = get_accuracy_level(["1.1.0.0"], ["1.1.1.1"])[0]
        assert sum(test3) == 2
        test4 = get_accuracy_level(["1.0.0.0"], ["1.1.1.1"])[0]
        assert sum(test4) == 1
        test5 = get_accuracy_level(["0.0.0.0"], ["1.1.1.1"])[0]
        assert sum(test5) == 0
        # Check that even with the 1 in the wrong place we still get the same answer
        test5 = get_accuracy_level(["2.1.1.1"], ["1.1.1.1"])[0]
        assert sum(test5) == 0
        test5 = get_accuracy_level(["2.2.1.1"], ["1.1.1.1"])[0]
        assert sum(test5) == 0
        test5 = get_accuracy_level(["2.2.2.1"], ["1.1.1.1"])[0]
        assert sum(test5) == 0

    def test_acc_multi_ec(self):
        values = get_accuracy_level(["1.1.1.1;1.1.1.0"], ["1.1.1.1"])[0]
        print(values)
        assert sum(values) == 4
        test2 = get_accuracy_level(["1.1.1.0;1.1.0.0"], ["1.1.1.1"])[0]
        assert sum(test2) == 3
        test3 = get_accuracy_level(["1.1.0.0;1.1.1.0"], ["1.1.1.1"])[0]
        assert sum(test3) == 3
        test4 = get_accuracy_level(["1.0.0.0;1.2.2.2"], ["1.1.1.1"])[0]
        assert sum(test4) == 1
        test5 = get_accuracy_level(["0.0.0.0;1.0.0.0;1.1.1.0"], ["1.1.1.1"])[0]
        assert sum(test5) == 3
        test6 = get_accuracy_level(["0.0.0.0;2.2.2.0"], ["1.1.1.1"])[0]
        assert sum(test6) == 0

    def test_mean_acc(self):
        # Now the goal is to check for the overall, make sure we group and get the maximum.
        cols = ['EC number', '0', '1', '2']
        rows = [['1.1.1.1', '1.1.1.1', '1.1.1.1', '1.1.1.1']]
        df = pd.DataFrame(rows, columns=cols)
        assert predict_k_accuracy(df, 1)[0] == 1
        assert predict_k_accuracy(df, 1)[1] == 1
        assert predict_k_accuracy(df, 1)[2] == 1
        assert predict_k_accuracy(df, 1)[3] == 1

        # Check another more complex case
        cols = ['EC number', '0', '1', '2']
        rows = [['1.1.1.1', '1.1.1.1', '1.1.1.1', '1.1.1.1'],
                ['0.0.1.1', '1.1.1.1', '1.1.1.1', '1.1.1.1']]
        df = pd.DataFrame(rows, columns=cols)
        assert predict_k_accuracy(df, 1)[0] == 0.5
        assert predict_k_accuracy(df, 1)[1] == 0.5
        assert predict_k_accuracy(df, 1)[2] == 0.5
        assert predict_k_accuracy(df, 1)[3] == 0.5

        # Check another more complex case
        cols = ['EC number', '0', '1', '2']
        rows = [['1.1.1.1', '1.1.1.1', '1.1.1.1', '1.1.1.1'],
                ['1.0.0.1', '1.1.1.1', '1.1.1.1', '1.1.1.1']]
        df = pd.DataFrame(rows, columns=cols)
        assert predict_k_accuracy(df, 1)[0] == 1.0
        assert predict_k_accuracy(df, 1)[1] == 0.5
        assert predict_k_accuracy(df, 1)[2] == 0.5
        assert predict_k_accuracy(df, 1)[3] == 0.5

        # Check another more complex case
        cols = ['EC number', '0', '1', '2']
        rows = [['1.1.1.1', '1.1.1.1', '1.1.1.1', '1.1.1.1'],
                ['1.0.0.1', '1.1.1.1', '1.1.1.1', '1.0.0.1']]
        
        df = pd.DataFrame(rows, columns=cols)
        assert predict_k_accuracy(df, 3)[0] == 1.0
        assert predict_k_accuracy(df, 3)[1] == 1.0
        assert predict_k_accuracy(df, 3)[2] == 1.0
        assert predict_k_accuracy(df, 3)[3] == 1.0
        
    def test_datasets(self):
        df = pd.read_csv(f'{output_dir}ProteInfer_70-90_protein_test_results_df.csv')
        print(predict_k_accuracy(df, 3))

if __name__ == '__main__':
    unittest.main()