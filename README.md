# CARE
CARE: Benchmarks for the Classification and Retrieval of Enzymes

## 1 env

```
#for BERC datasets, splitting, and analysis
conda create -n BERC_analysis python=3.8 -y
conda activate BERC_analysis
conda install -c rdkit rdkit=2020.03.3 -y
#conda install -c tmap tmap -y
#pip install rxnfp
pip install -U scikit-learn
#pip install simpletransformers==0.61.13
pip install simpletransformers "transformers==4.30.2"
pip install seaborn

#BERC benchmarking is done through other pacakges
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