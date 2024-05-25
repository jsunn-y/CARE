# CARE
CARE: Benchmarks for the Classification and Retrieval of Enzymes

## Installation

```
git clone https://github.com/jsunn-y/CARE/
cd CARE

#for CARE dataset generation, splitting, BLAST, visualization
#only install this environment if you want to reproduce the steps used to generate the datasets and splits in this work
conda create -n CARE_processing python=3.8 -y
conda activate CARE_processing
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c conda-forge -c bioconda mmseqs2
pip install scipy pandas seaborn npysearch
pip install seaborn

#for CREEP model training and evaluation
#only install this environment if you would like to run training and inference with CREEP
cd CREEP
conda create -n CREEP python=3.8
conda activate CREEP
pip install pandas torch==2.2.0 transformers==4.39.1 sentencepiece
pip install -e .
#pip install lxml #doesn't look like you need this

#instructions for CARE benchmarking using other packages is provided in more detail in the sections below
```
## Dataset curation and splitting
Code used to generates the datasets and splits for this work can be found in the jupyter notebooks in `generate_dataset_splits` with an overview [here](generate_datasets_splits).

The outputs from these notebooks include the complete datasets found in `processed_data` and the train and test splits found in `splits`.

The table below summarizes which files should be used for each train-test split described in the work.

## Benchmarking
Accuracy metrics for benchmarking can be obtained and visualized using `analysis.ipynb`. Required format for analysis of one baseline is a csv file where each row is part of the test set, and each row is associated with a ranking of EC numbers ranked from best to worst. An example of this file is: 

Accuracy analysis can be performed in most environments with minimal packages. Additional performance metrics besides k=1 accuracy from the paper can be found in this notebook. 

## Baselines for task 1 (protein to EC/reaction classification)


## Baselines for task 2 (reaction to EC/protein retrieval)

Outputs from CREEP, CLIPZyme, and the Similarity Baseline will outputting in the format of npy files containing arrays of representations. A similarity search can be performed to obtain a ranking of EC numbers, using `task2_baselines/tabulate_results.ipynb` The outputs will be csv files saved to their respective folders, to be used for performance analysis. Refer to each model below for details on their specific implementation. 
