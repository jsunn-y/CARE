## Baselines for task 1 (protein to EC/reaction classification)

Note: this workflow is still under construction. The results of EC classification for each method and each split is found in `task1_baselines/results_summary` as .csv files, for use in downstream performance analysis. Task 1 benchmarking (excluding CLEAN) can be reproduced with a single command using the CARE package.

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

### BLAST
```
CARE task1 --baseline BLAST --query-dataset 30 --k 10 --pretrained-dir CARE_PATH --output-dir OUTPUT_PATH
```
Where CARE_PATH is the folder where you downloaded the CARE github repository (and saved the data in the folder `processed_data` from zeonodo), Our retrained CLEAN  model is available at [CARE_pretrained.zip]([https://zenodo.org/records/12195378](https://zenodo.org/records/12207966)).
Where `baseline` is one of "All", "BLAST", "ChatGPT", "ProteInfer", or "Random" and `query_dataset` is  one of "All", "30", "30-50", "Price", or "promiscuous". Note `--pretrained-dir` is the path to the CARE directory on your computer where you have downloaded the data that we have provided or generated your own datasets using the same approach.
### ProteInfer
For proteInfer, you need to first download the proteInfer directory into the proteInfer folder (https://github.com/google-research/proteinfer) into `task1_baselines/ProteInfer/proteinfer/` and create a `proteinfer` environement as per the proteinfer github. Then you can run the following command:
```
CARE task1 --baseline ProteInfer --query-dataset 30 --k 10 --pretrained-dir CARE_PATH --output-dir OUTPUT_PATH
```

### Random
```
CARE task1 --baseline Random --query-dataset 30 --k 10 --pretrained-dir CARE_PATH --output-dir OUTPUT_PATH
```

### CLEAN
For CLEAN follow the specific notebook (`CARE_forCLEAN.ipynb`) as this requires more input.
```
CARE task1 --baseline CLEAN --query-dataset 30 --k 10 --pretrained-dir CARE_PATH --output-dir OUTPUT_PATH
```

### ChatGPT
```
CARE task1 --baseline ChatGPT --query-dataset 30 --k 10 --pretrained-dir CARE_PATH --output-dir OUTPUT_PATH
```
Note for chatGPT, you'll need to make sure your API key is saved in a secrets.txt file in the folder where you're running this from.



To get help: `CARE task1 --help`

### CLEAN
Outputs from model training and inference in our study are found in `CLEAN`. Alternatively, instructions for reproducing our retraining and inference procedure can be found in `task1_baselines/CLEAN/CARE_forCLEAN.ipynb`, which is performed within the [CLEAN package](https://github.com/tttianhao/CLEAN/tree/main). For training, we did not perform any clustering, and we used the recommended training parameters with triplet margin loss.

Our retrained CLEAN  model is available at [CARE_pretrained.zip]([https://zenodo.org/records/12195378](https://zenodo.org/records/12207966)).
