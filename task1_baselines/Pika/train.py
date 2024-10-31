import sys
from pika.main import Pika
from pika.utils.helpers import load_config
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
from pika.main import Pika
from pika.utils.helpers import load_config
import warnings
import logging
import pandas as pd
import numpy as np

# Make the model 
# prep config
assets_path = "../assets/"
config = load_config("pika_config.json")
config["datamodule"]["split_path"] = "split.csv"
model = Pika(config)
model.train()

# For each of the test sets we want to 
splits = ['30', '30-50', 'price', 'promiscuous']
rows = []
for split in splits: 
    df_test = pd.read_csv(f'../../splits/task1/{split}_protein_test.csv')
    
    for entry, seq in df_test[['Entry', 'Sequence']].values:
        ec = model.enquire(
            proteins=seq,
            question="What is the EC number of this protein?"
        )
        rows.append([split, seq, entry, '|'.join(ec)])
saving_df = pd.DataFrame(rows, columns=['Split', 'seq', 'Entry', 'EC'])
saving_df.to_csv(f'all_test_datasets_output_EC-number.csv', index=False)

### Save the results now individually 

df = saving_df.copy()

# The datasets we want to go through
splits = ['30', '30-50', 'price', 'promiscuous']

for split in splits:
    
    # Entry,EC number,
    sub_df = df[df['Split'] == split]
    # Make the enrty to the EC 
    test_df = pd.read_csv(f'../../splits/task1/{split}_protein_test.csv')

    # Make sure the EC is clean
    sub_df['EC number'] = [e.strip() for e in sub_df['EC'].values]
    
    # Make the EC format the same as the other datasets
    entry_to_ec = dict(zip(sub_df['Entry'], sub_df['EC number']))
    test_df['0'] = [entry_to_ec.get(e) for e in test_df['Entry'].values]
    
    test_df.to_csv(f'../results_summary/Pika/{split}_protein_test_results_df.csv', index=False)
