import sys
#sys.path.append('/disk1/ariane/vscode/CARE/task1_baselines/Pika/Pika')
from pika.main import Pika
from pika.utils.helpers import load_config
import warnings
import logging
import pandas as pd

df_train = pd.read_csv('/disk1/ariane/vscode/CARE/splits/task1/protein_train.csv')
rows = []
for entry, seq, ec in df_train[['Entry', 'Sequence', 'EC number']].values:
    rows.append([entry, 'qa', f"What is the EC number of this protein? {ec}"])
    
sample_annotations = pd.DataFrame(rows, columns=['uniprot_id', 'type', 'annotation'])
sample_annotations.to_csv('annotations.csv', index=False)

# Also split into a train test and validation set for the model training
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_train, test_size=0.3)
rows = []
for entry, seq, ec in df_train[['Entry', 'Sequence', 'EC number']].values:
        rows.append([entry, len(seq), 'train'])
    
for entry, seq, ec in test[['Entry', 'Sequence', 'EC number']].values[:int(0.5*(len(test)))]:
    rows.append([entry, len(seq), 'test'])

for entry, seq, ec in test[['Entry', 'Sequence', 'EC number']].values[int(0.5*(len(test))):]:
    rows.append([entry, len(seq), 'val'])
    
sample_split = pd.DataFrame(rows, columns=['uniprot_id' , 'protein_length', 'split'])

sample_split.to_csv('split.csv', index=False)

# Next we need to make the metrics
# uniprot_id,metric,value
# A0A068BGA5,is_enzyme,True

from sklearn.model_selection import train_test_split
# A0A084R1H6,in_membrane,False
# A0A084R1H6,in_nucleus,False
# A0A084R1H6,in_mitochondria,False
# A0A084R1H6,is_enzyme,True
# A0A084R1H6,mw,263256
rows = []
for entry, seq, ec in df_train[['Entry', 'Sequence', 'EC number']].values[100:]:
    rows.append([entry, 'is_enzyme', True])
    rows.append([entry, 'in_membrane', True])
    rows.append([entry, 'in_mitochondria', True])
    rows.append([entry, 'in_nucleus', True])
    rows.append([entry, 'cofactor', "heme"])
    rows.append([entry, 'localization', "heme"])
    rows.append([entry, 'mw', 263256])
    
for entry, seq, ec in df_train[['Entry', 'Sequence', 'EC number']].values[:100]:
    rows.append([entry, 'is_enzyme', False])
    rows.append([entry, 'in_membrane', False])
    rows.append([entry, 'in_mitochondria', False])
    rows.append([entry, 'in_nucleus', False])
    rows.append([entry, 'localization', "heme"])
    rows.append([entry, 'mw', 23256])

sample_metrics = pd.DataFrame(rows, columns=['uniprot_id' , 'metric', 'value'])
sample_metrics.to_csv('metrics.csv', index=False)


# Save the training sequences
rows = []
for entry, seq, ec in df_train[['Entry', 'Sequence', 'EC number']].values:
    rows.append([entry, '', '', seq, len(seq), 1, 1, 1, 1])
    
sample_seqs = pd.DataFrame(rows, columns=['uniprot_id', 'uniref_cluster', 'taxonomy', 'sequence', 'length', 'mw', 'num_fields', 'num_summary', 'num_qa'])

sample_seqs.to_csv('sequences.csv', index=False)

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
    df_test = pd.read_csv(f'/disk1/ariane/vscode/CARE/splits/task1/{split}_protein_test.csv')
    
    for entry, seq in df_test[['Entry', 'Sequence']].values:
        ec = model.enquire(
            proteins=seq,
            question="What is the EC number of this protein?"
        )
        rows.append([split, seq, entry, '|'.join(ec)])
saving_df = pd.DataFrame(rows, columns=['Split', 'seq', 'Entry', 'EC'])
saving_df.to_csv('output_Everything.csv', index=False)
