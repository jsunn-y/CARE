# CARE:  a Benchmark Suite for the Classification and Retrieval of Enzymes
CARE is a datasets and benchmarks suite to evaluate the performance of models to predict the functions of enzymes. CARE is split into two tasks: classification of enzyme sequences based on Enzyme Commission (EC) number (Task 1), and retrieval of EC number given a reaction (Task 2).

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

## Performance Evaluation
After ranking EC numbers from best to worst, performance metrics for benchmarking can be obtained using `performance_evaluation.ipynb`. Required format for analysis of each model on each split is a .csv file where each row is an entry in the test set, and each entry is associated with a ranking of EC numbers ranked from best to worst ([example](link)).

Performance analysis can be performed in most environments with minimal packages. The standard performance metric is k=1 classification/retrieval accuracy, but we also provide code to calculate other metrics in this notebook. The output for k=1 accuracy should look something like this:
| Method | Level 4 Accuracy (X.X.X.X) | Level 3 Accuracy (X.X.X.-) | Level 2 Accuracy (X.X.-.-) | Level 1 Accuracy (X.-.-.-) |
|:--------|:-------:|:-------:|:-------:|:-------:|
| Name |  54.1 | 60.4 | 81.0 | 95.5 |
| ... | ... | ...|  ...|  ... |

## Benchmarking pretrained models

### Installation
If you are only interested in using the datasets and train-test splits in CARE, skip the installation steps below and directly download the data from [here](link). If you are interested in the reproducing the dataset curation/splitting, training and inference, and analyses in our study, then proceed to clone this repo and install the relevant environments below:

```
git clone https://github.com/jsunn-y/CARE/
cd CARE
conda create -n CARE_processing python=3.8 -y

conda activate CARE_processing
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c conda-forge -c bioconda mmseqs2 -y
pip install dist/care.0.0.1.tar.gz
```

The outputs from pretrained models are provided [here](link).

### Task 1: performance evaluation 
Task 1 benchmarks (excluding CLEAN) can be reproduced with a single command using the CARE package,  
** Note **: install this environment before running training and inference with ProteInfer.
```
conda create --name proteinfer python=3.7 -y
conda activate proteinfer
git clone https://github.com/google-research/proteinfer
cd ~/proteinfer
pip3 install -r requirements.txt
```

Run the following command to make predictions using pretrained models and automatically find the most likely ECs and calculate accuracy metrics.
```
CARE task1 --baseline All --query-dataset All --k 10 
```
Where `baseline` is one of "All", "BLAST", "ChatGPT", "ProteInfer", or "Random" and `query_dataset` is  one of "All", "30", "30-50", "Price", or "promiscuous".  

To get help: `CARE task1 --help`

Detailed instructions for reproducing our baselines on Task 1 and general recommendations for benchmarking on Task 1 can be found [here](task1_baselines). CLEAN requires retraining the model, which is explained in detail.

### Task 2: performance evaluation 
To perform the standard benchmarking using our pretrained model outputs, download the data from [here](link) and replace the folder `task2_baselines`. Then you can run the CARE package any of the results for a specific tool or split:
```
CARE task2 --baseline All --query-dataset All
```
Where `baseline` is one of "All", "Similarity", "CREEP", "CREEP_text", "CLIPZyme", and "Random". Query dataset is one of "All", "easy", "medium" or "hard".

To get help: `CARE task2 --help`

Detailed instructions for reproducing our baselines on Task 2 and general recommendations for benchmarking on Task 2 can be found [here](task2_baselines).

## Task 1 methods installation:



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

**Note** you need to have downloaded the data and placed the data folder in the CARE directory. 

## Task 2 installation:

#### ChatGPT
For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line. This requires your OpenAI account to have an assoiated API key.

### Other information
Instructions for CARE benchmarking using other packages is provided in more detail in the sections below.

## Details on Task 1


## Details on Task 2



### CREEP
We introduce Contrastive Reaction-EnzymE Pretraining (CREEP), which is one of the first models that can perform Task 2 by aligning representations from different modalities (reaction, protein, and optionally textual description). The model is found under `CREEP`, while example usage is found under `task2_baselines/CREEP`.

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

#### Development

To build:
```
python setup.py sdist bdist_wheel
```

To install: 

```
pip install dist/care-0.1.0.tar.gz
```
