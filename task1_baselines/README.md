
## Baselines for task 1 (protein to EC/reaction classification)

Note: this workflow is still under construction. The results of EC classification for each method and each split is found in `task1_baselines/results_summary` as .csv files, for use in downstream performance analysis. For each method in task 1, you can generate an output csv of EC classifications by running a single script or following instructions provided in a corresponding notebook.

### Random
Run the following script:
```
python rank_tabulate_random.py
```

### BLAST
Run the following script:
```
python BLAST/run_diamond_blast.py
```

### ChatGPT
** Note **: For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line, from your OpenAI account.

### CLEAN
Outputs from model training and inference in our study are found in `CLEAN`. Alternatively, instructions for reproducing our retraining and inference procedure can be found in `task1_baselines/CLEAN/CARE_forCLEAN.ipynb`, which is performed within the [CLEAN package](https://github.com/tttianhao/CLEAN/tree/main). For training, we did not perform any clustering, and we used the recommended training parameters with triplet margin loss.

Our retrained CLEAN  model is available on [huggingface](https://huggingface.co/jsunn-y/CARE_pretrained).

### Pika
The pretrained model checkpoint is available on [huggingface](https://huggingface.co/jsunn-y/CARE_pretrained).
