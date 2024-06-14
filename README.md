# CARE:  a Benchmark Suite for the Classification and Retrieval of Enzymes
CARE is a datasets and benchmarks suite to evaluate the performance of models to predict the functions of enzymes. CARE is split into two tasks: classification of enzyme sequences based on Enzyme Commission (EC) number (Task 1), and retrieval of EC number given a reaction (Task 2).

## Installation
If you are only interested in using the datasets and train-test splits in CARE, skip the installation steps below and directly download the data from [here](link). If you are interested in the reproducing the dataset curation/splitting, training and inference, and analyses in our study, then proceed to clone this repo and install the relevant environments below:

## CARE benchmarking

```
git clone https://github.com/jsunn-y/CARE/
cd CARE
conda create -n CARE_processing python=3.8 -y

conda activate CARE_processing
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c conda-forge -c bioconda mmseqs2 -y
pip install dist/care.0.0.1.tar.gz
```

### Running CARE

### Task 1: performance evaluation 
To perform the standard benchmarking using our pretrained models and the splits provided download the data from [here](link) and put it in a folder (we suggest the name pretrained). Then you can run any of the results for a specific tool or split:

e.g. to run for BLAST on the 30% split you would run:
```
care benchmark --task 1 --baseline BLAST --query_dataset blast --output_folder "path_to_output"
```

### Task 1: prediction of EC numbers from protein
To use the task1 pretrained results to query a specific protein sequence with a specific method:

```
care benchmark --task 1 --baseline BLAST --query_protein "MASMSMAAAM" --output_folder "path_to_output"
```


### Task 2: performance evaluation 
To perform the standard benchmarking using our pretrained models and the splits provided download the data from [here](link) and put it in a folder (we suggest the name pretrained). Then you can run any of the results for a specific tool or split:

e.g. to run for Similarity searching on the easy split you would run:
```
care benchmark --task 2 --baseline Similarity --query_dataset easy --output_folder "path_to_output"
```

### Task 1: prediction of EC numbers from a reaction
You can use the pretrained models to also query a speicfic reaction and obtain the EC number for that reaction
```
care benchmark --task 2 --baseline Similarity --query_recation "CC(C)=CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)C" --output_folder "path_to_output"
```

### Task 1: prediction of EC numbers from protein
To use the task1 pretrained results to query a specific protein sequence with a specific method:

```
care benchmark --task 1 --baseline BLAST --query_protein "MASMSMAAAM" --output_folder "path_to_output"
```


## CARE development and retraining

## Task 1 

#### ProteInfer installation
For ProteInfer model inference.
** Note **: only install this environment if you would like to run training and inference with ProteInfer.

```
conda create --name proteinfer python=3.7 -y
conda activate proteinfer
git clone https://github.com/google-research/proteinfer
cd ~/proteinfer
pip3 install -r requirements.txt
```

#### CLEAN installation
For CLEAN model inference.
** Note **: only install this environment if you would like to run training and inference with CLEAN.
```
conda create -n clean python==3.10.4 -y
conda activate clean
pip install -r clean_requirements.txt
```

#### ChatGPT
For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line. This requires your OpenAI account to have an assoiated API key.

## Running:

To run task 1, simply run (from the CARE folder):
```
python task1.py --split "30" --tool "BLAST" --outputdir "a_path"
```
Where tool is one of "BLAST", "ChatGPT", "ProteInfer", "CLEAN", or "Random".

**Note** you need to have downloaded the data and placed the data folder in the CARE directory. 

## Task 2 installation:

#### CREEP installation
For CREEP model training and inference
** Note **: only install this environment if you would like to run training and inference with CREEP.

```
cd CREEP
conda create -n CREEP python=3.8
conda activate CREEP
#we recommend installing this way so that torch is compatible with your GPU and your version of CUDA
pip install pandas torch==2.2.0 transformers==4.39.1 sentencepiece
pip install -e .
#pip install lxml #doesn't look like you need this
```

#### ChatGPT
For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line. This requires your OpenAI account to have an assoiated API key.

### Other information
Instructions for CARE benchmarking using other packages is provided in more detail in the sections below.

## Datasets and splits
Processed datasets/splits should should be downloaded from [here](link) to replace the empty folders `processed_data` and `splits`, respectively. Note that in the full datasets and train sets, every row represents a unique protein-EC pair, or a unique reaction-EC pair. In the test sets, every row is also a unique protein-EC or reaction-EC pair, but for the promiscuous test set, each row maps a protein seqeunce to a list of corresponding ECs.

The table below summarizes which files should be used for each train-test split described in the work.

| Task | Split |Train File | Test File | Optional Train Files |
|:--------|:-------:|:-------:|:-------:|:-------:|
| Task 1 | <30% Identity | `protein_train.csv` | `30_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|
|  | 30-50% Identity | `protein_train.csv` | `30-50_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|
|  | 50-70% Identity | `protein_train.csv` | `50-70_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|
|  | 70-90% Identity | `protein_train.csv` | `70-90_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|
|  | Misclassified (Price) | `protein_train.csv` | `price_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|
|  | Promiscuous | `protein_train.csv` | `promiscuous_protein_test.csv` | `reaction2EC.csv` `text2EC.csv`|

| Task | Split |Train File | Test File |  Optional Train Files |
|:--------|:-------:|:-------:|:-------:|:-------:| 
| Task 2 |  Easy | `easy_reaction_train.csv` | `easy_reaction_test.csv` | `protein2EC.csv` `text2EC.csv`|
|  | Medium | `medium_reaction_train.csv` | `medium_reaction_test.csv` |  `protein2EC.csv` `text2EC.csv`|
|  | Hard | `hard_reaction_train.csv` | `hard_reaction_test.csv` |  `protein2EC.csv` `text2EC.csv`|

Alteratively, the steps used to generate the datasets and splits for this work can be reproduced using the jupyter notebooks in `generate_dataset_splits` with an overview [here](generate_datasets_splits).

## Trained models and representations
We provide the pretrained or retrained models for CREEP, CLEAN, ProteInfer, ClIP-Zyme and drfp encodings in the `models` folder. Each tool has an associated folder and that is the model (and or code) used in our benchmarking paper and the defaults that are used by the CARE package. We expect the `models` folder to be located within the `data` folder. 

## Performance Evaluation
After training, performance metrics for benchmarking can be obtained using `performance_evaluation.ipynb`. Required format for analysis of each model on each split is a .csv file where each row is an entry in the test set, and each entry is associated with a ranking of EC numbers ranked from best to worst ([example](link)).

Performance analysis can be performed in most environments with minimal packages. The standard performance metric is k=1 classification/retrieval accuracy, but we also provide code to calculate other metrics in this notebook. The output for k=1 accuracy should look something like this:
| Method | Level 4 Accuracy (X.X.X.X) | Level 3 Accuracy (X.X.X.-) | Level 2 Accuracy (X.X.-.-) | Level 1 Accuracy (X.-.-.-) |
|:--------|:-------:|:-------:|:-------:|:-------:|
| Name |  54.1 | 60.4 | 81.0 | 95.5 |
| ... | ... | ...|  ...|  ... |


## Baselines for task 1 (protein to EC classification)
Detailed instructions for reproducing our baselines on Task 1 and general recommendations for benchmarking on Task 1 can be found [here](task1_baselines).

## Baselines for task 2 (reaction to EC retrieval)

Detailed instructions for reproducing our baselines on Task 2 and general recommendations for benchmarking on Task 2 can be found [here](task2_baselines).

### CREEP
We introduce Contrastive Reaction-EnzymE Pretraining (CREEP), which is one of the first models that can perform Task 2 by aligning representations from different modalities (reaction, protein, and optionally textual description). The model is found under `CREEP`, while example usage is found under `task2_baselines/CREEP`.
