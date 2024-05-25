## Baselines for task 1 (protein to EC/reaction classification)

The required CSV for performance analysis using can be obtained for each method as follows:

### BLAST

### CLEAN
Instructions for retraining and performing inference with CLEAN can be found in `task1_baselines/CLEAN/CARE_forCLEAN.ipynb` Outputs from model training and inference are found in `task1_baselines/CLEAN`. For training, we did not perform any clustering, and we used the recommended training parameters with triplet margin loss.

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
