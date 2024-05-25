## Baselines for task 1 (protein to EC/reaction classification)

The results of EC classification for each method and each split is found in `results_summary` as .csv files, for use in downstream performance analysis. To reproduce the steps to generate each classification result for each method explored in this study:

### BLAST

### CLEAN
Outputs from model training and inference in our study are found in `CLEAN`. Alternatively, instructions for reproducing our retraining and inference procedure can be found in `task1_baselines/CLEAN/CARE_forCLEAN.ipynb`, which is performed within the [CLEAN package](https://github.com/tttianhao/CLEAN/tree/main). For training, we did not perform any clustering, and we used the recommended training parameters with triplet margin loss.

### ProtInfer

https://github.com/google-research/proteinfer

```
conda create --name proteinfer python=3.7 -y
conda activate proteinfer
git clone https://github.com/google-research/proteinfer
cd ~/proteinfer
pip3 install -r requirements.txt
```


### Pika
Protein language model querying.
```
yes | conda create --name pika python=3.10
pip install git+https://github.com/EMCarrami/Pika.git
```
