import os
import pandas as pd
from blast_utils import make_diamond_db, diamond_alignment

os.makedirs('ref_databases', exist_ok=True)
#make diamond reference database for BLAST
make_diamond_db('../../splits/task1/fastas/train_swissprot.fasta', 'ref_databases/train_swissprot') 

#mapping of reference IDs to EC numbers
ref_withEC = pd.read_csv('../../processed_data/protein2EC.csv')
entry2EC = dict(zip(ref_withEC['Entry'], ref_withEC['EC All']))


names= ['price_protein_test', 'promiscuous_protein_test', '30_protein_test', '30-50_protein_test']

os.makedirs('../results_summary/BLAST', exist_ok=True)

for name in names:

    df = pd.read_csv(f'../../splits/task1/{name}.csv')
    results = diamond_alignment(f'../../splits/task1/fastas/{name}.fasta', 'ref_databases/train_swissprot')
    #turn indices into a column
    results['Entry'] = results.index
    #merge the two dataframes
    merged = pd.merge(df, results, on='Entry', how='left')
    #map to predicted EC number
    merged['EC Predicted'] = merged['ref_entry_id'].map(entry2EC)

    merged = merged.rename(columns={'EC Predicted': 0})
    merged[['Entry', 'EC number', 0]].to_csv(f'../results_summary/BLAST/{name}_results_df.csv', index=False)