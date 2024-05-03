# CARE
CARE: Benchmarks for the Classification and Retrieval of Enzymes

## Installation

```
git clone https://github.com/jsunn-y/CARE/
cd CARE

#for CARE datasets, splitting, and analysis
conda create -n CARE_processing python=3.8 -y
conda activate CARE_processing
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c conda-forge -c bioconda mmseqs2
pip install scipy pandas seaborn

#CARE benchmarking is done through other pacakges
#BLAST

#for CREEP model training and evaluation
cd CREEP
conda create -n CREEP python=3.8
conda activate CREEP

pip install pandas torch==2.2.0 transformers==4.39.1 sentencepiece
pip install -e .
#pip install lxml #doesn't look like you need this
```
## Dataset curation and splitting

## Baselines for task 1 (protein to EC/reaction classification)

### BLAST

### CLEAN

## Baselines for task 2 (reaction to EC/protein retrieval)

### CREEP
We propose Contrastive Reatction-EnzymE Pretraining, as summaried in our mansucript. CREEP training and retrieval is performed with three steps: 
(1) contrastive representation alignment by finetuning lanugage models from different modalities, (2) extraction of protein and reaction representations using the finetuned models, and (3) retrieval of proteins using a similarity search in the embedding space.

For example, for one of the splits.

1. Go to the folder `task2_baselines/CREEP/`. Set the output directory with `export OUTPUT_DIR=output/easy_split`. Run finetuning training with default parameters:
```
python step_01_train_CREEP.py --output_model_dir="$OUTPUT_DIR" --train_split=easy_reaction_train
```
Or
```
python step_01_train_CREEP.py --output_model_dir=output/easy_split --train_split=easy_reaction_train
```

Note that our batch size of 16 is optimized for a single 80GB GPU. Training for 30 epochs took about 18 hrs on a single H100 GPU. Training outputs will be saved in the ouput directory.

2. For extracting the reference protein representations and their cluster centers: 
```
python step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=all_proteins --modality=protein --get_cluster_centers
```
Note that this will take 0.5-1 hours on a single H100 GPU.

For extracting the query reaction representations for each test set: 
```
python step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=easy_reaction_test --modality=reaction
python step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=easy_reaction_test --modality=text
```
Representations will be svaed in the output directory under `representations`.

3. Run the retrieval similarity search:
```
python step_03_downstream_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
python step_03_downstream_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=text --reference_modality=protein
```
The outputs will be saved under `rettieval_results` and can be further analyzed and visualized in `retrieval_analysis_metrics.ipynb`.
### CLIPZyme
