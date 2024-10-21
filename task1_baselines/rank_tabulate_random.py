import numpy as np
import pandas as pd
import random
import os

for baseline in  ['random']:
    for split in ['30', '30-50', 'price', 'promiscuous']:
        np.random.seed(42)
        num_ecs = 50
        rows = []
        df = pd.read_csv('../splits/task1/{}_protein_test.csv'.format(split))
        train_ecs = set(pd.read_csv(os.path.join('../processed_data/reaction2EC.csv'))['EC number'].values)

        for entry, seq, true_ecs in df[['Entry', 'Sequence', 'EC number']].values:
            # Randomly sample without replacement an entry from the training df
            predicted_ec = random.sample(train_ecs, k=num_ecs)
            rows.append([entry, seq, true_ecs] + predicted_ec)

        new_df = pd.DataFrame(rows)
        new_df.columns = ['Entry', 'Sequence', 'EC number'] +  [str(s) for s in range(0, num_ecs)]

        os.makedirs('results_summary/{}'.format(baseline), exist_ok=True)

        new_df.to_csv('results_summary/{}/{}_protein_test_results_df.csv'.format(baseline,split), index=False)