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
