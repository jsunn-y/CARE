## Baselines for task 1 (protein to EC/reaction classification)

The results of EC classification for each method and each split is found in `task1_baselines/results_summary` as .csv files, for use in downstream performance analysis. Task 1 benchmarking (excluding CLEAN) can be reproduced with a single command using the CARE package.

** Note **: install this environment before running any training and inference with ProteInfer.
```
conda create --name proteinfer python=3.7 -y
conda activate proteinfer
git clone https://github.com/google-research/proteinfer
cd ~/proteinfer
pip3 install -r requirements.txt
```
** Note **: For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line, from your OpenAI account.

Run the following command to make predictions using pretrained models and automatically find the most likely ECs and calculate accuracy metrics.
```
CARE task1 --baseline All --query-dataset All --k 10 
```
Where `baseline` is one of "All", "BLAST", "ChatGPT", "ProteInfer", or "Random" and `query_dataset` is  one of "All", "30", "30-50", "Price", or "promiscuous".  

To get help: `CARE task1 --help`


### CLEAN
Outputs from model training and inference in our study are found in `CLEAN`. Alternatively, instructions for reproducing our retraining and inference procedure can be found in `task1_baselines/CLEAN/CARE_forCLEAN.ipynb`, which is performed within the [CLEAN package](https://github.com/tttianhao/CLEAN/tree/main). For training, we did not perform any clustering, and we used the recommended training parameters with triplet margin loss.
