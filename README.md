# CARE
CARE: Benchmarks for the Classification and Retrieval of Enzymes

## 1 env

```
#for CARE datasets, splitting, and analysis
conda create -n CARE_processing python=3.8 -y
conda activate CARE_processing
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c conda-forge -c bioconda mmseqs2
pip install pandas
pip install seaborn

#CARE benchmarking is done through other pacakges
#BLAST

#for CREEP model training and evaluation
conda create -n CREEP python=3.8
conda activate CREEP
pip install pandas
#conda install -y numpy networkx scikit-learn
pip install torch==2.2.0 
pip install transformers==4.30.2 #might need a newer version 
pip install sentencepiece
pip install lxml
pip install -e .
```
## Dataset curation and splitting

## Baselines for task 1 (protein to EC/reaction classification)

### BLAST

### CLEAN

## Baselines for task 2 (reaction to EC/protein retrieval)

### CREEP
We propose Contrastive Reatction-EnzymE Pretraining, as summaried in our mansucript. CREEP training and retrieval is performed with three steps: 
(1) contrastive representation alignment by finetuning lanugage models from different modalities, (2) extraction of protein and reaction representations using the finetuned models, and (3) retrieval of proteins using a similarity search in the embedding space.

Run finetuning training with default parameters:
```
python step_01_train_CREEP.py --output_model_dir=output/default
```
Note that our batch size of 16 is optimized for a single 80GB GPU. Training for 30 epochs took about 18 hrs on a single H100 GPU.



### CLIPZyme
