# BERC
BERC: Benchmarks for Enzyme Retrieval and Classification

## 1 env

```
conda create -n ProteinDT python=3.8
conda activate ProteinDT

conda install -y numpy networkx scikit-learn

pip install torch==1.10.* #torch 2.2.0 and CUDA 12.1 also works

pip install transformers # had to use version 4.30.2 to be compatible with rxnfp (newer models need 4.39.1)
pip install sentencepiece
pip install lxml

# for TAPE
pip install lmdb
pip install seqeval

# for baseline ChatGPT
pip install openai

# for baseline Galactica
pip install accelerate

# for visualization
pip install matplotlib

# for binding editing
pip install h5py
pip install torch_geometric==2.0 torch_scatter torch_sparse torch_cluster #torch geometric 2.5 also works
pip install biopython

# for ESM folding
#pip install "fair-esm[esmfold]"
#pip install dm-tree omegaconf ml-collections einops
#pip install fair-esm[esmfold]==2.0.0  --no-dependencies # Override deepspeed==0.5 
#pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
#pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

#conda install mdtraj biopython -c conda-forge -yq

#for similarity search
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl

#for rxnfp
conda create -n rxnfp python=3.8 -y
conda activate rxnfp
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c tmap tmap -y
pip install rxnfp
pip install -U scikit-learn
#pip install simpletransformers==0.61.13
pip install simpletransformers "transformers==4.30.2"

# for ProteinDT
pip install -e .
```
