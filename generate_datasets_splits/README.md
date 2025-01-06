`rawdata_processing.ipynb`: Instructions to download all of the raw data used in this study and steps to process the raw data. Clustering is also performed using MMseqs2 at 30, 50, 70, and 90% identity, and saved to the processed dataset for later use. Outputs are saved to `../processed_data` and include files such as `protein2EC.csv`, `reaction2EC.csv`, and `text2EC.csv`. Alternatively, raw data can be downloaded from [CARE_raw_data.zip](https://zenodo.org/records/14004425) and should be uploaded to `raw_data`. 

`splitting_task1.ipynb`: Steps to generate the train-test splits for Task 1, from the processed data. All outputs are saved to `../splits/task1`

`splitting_task2.ipynb`: Steps to generate the train-test splits for Task 2, from the processed data. All outputs are saved to `../splits/task2`

`dataset_visualization.ipynb`: Notebook to visualize the distribution of the processed datasets and splits.


