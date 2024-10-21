import numpy as np
import pandas as pd
import os
from drfp import DrfpEncoder

for split in ['easy', 'medium', 'hard']:

    query_df = pd.read_csv('../../splits/task2/{}_reaction_test.csv'.format(split))
    query_reactions = query_df['Reaction'].values

    fps = DrfpEncoder.encode(query_reactions, show_progress_bar=True)

    fps = np.vstack(fps)
    os.makedirs('output/{}_split/representations/'.format(split), exist_ok=True)

    saved_file_path = os.path.join('output/{}_split/representations/{}_reaction_test_representations'.format(split, split))

    #if the file exists, load it
    if os.path.exists(saved_file_path + ".npy"):
        results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
    else:
        results = {}

    results["reaction_repr_array"] = fps
        
    np.save(saved_file_path, results)

#only uses one cpu, should take about 12 minutes per split
for split in ['easy', 'medium', 'hard']:
    output_folder = 'output/{}_split/representations/'.format(split)
    
    os.makedirs('output/{}_split/representations/'.format(split), exist_ok=True)

    df = pd.read_csv('../../splits/task2/{}_reaction_train.csv'.format(split))

    reactions = df['Reaction'].values

    fps = DrfpEncoder.encode(reactions, show_progress_bar=True)
    repr_array = np.vstack(fps)

    df['index'] = df.index
    ec2index = df.groupby('EC number')['index'].apply(list).to_frame().to_dict()['index']
    EClist = np.loadtxt("../../processed_data/EC_list.txt", dtype=str)

    print(len(EClist))
    print(len(ec2index.keys()))

    #temporary line to just check if the code runs
    #EClist = [ec for ec in EClist if ec in ec2index.keys()]
    #assert len(EClist) == len(ec2index.keys())
    
    cluster_centers = np.zeros((len(EClist), repr_array.shape[1]))
    for i, ec in enumerate(EClist):
        #average together the embeddings for each EC number
        if ec in ec2index.keys():
            indices = ec2index[ec]
            cluster_centers[i] = np.mean(repr_array[indices], axis=0)

    saved_file_path = os.path.join(output_folder, "all_ECs_cluster_centers")
    #if the file exists, load it
    if os.path.exists(saved_file_path + ".npy"):
        results = np.load(saved_file_path + ".npy", allow_pickle=True).item()
    else:
        results = {}

    results["reaction_repr_array"] = cluster_centers

    print(cluster_centers.shape)
    np.save(saved_file_path, results)