## Baselines for task 2 (reaction to EC/protein retrieval)
Note: this workflow is still under construction. The results of EC retrieval for each method and each split is found in `results_summary` as .csv files, for use in downstream performance analysis. 

Alternatively, these results can be reproduced at a high level (excluding ChatGPT and Random) by following these steps: 

1. Model trainining for CREEP and CLIPZyme. Similarity baseline can skip to step 2.
2. Extract representations in the format of .npy files containing arrays of representations from multiple modalities (such as protein, reaction, and text).
3. A similarity search between the train and test set is performed using `downstream_retrieval.py`. Retrieval similarites are outputed as .npy arrays under the respective method and model folder in `retrieval_results`.
4. The retrieval similarities are processed to obtain a ranking of EC numbers using `rank_tabulate_similarities.oy`. The outputs will be .csv files saved to their respective folders in `results_summary`, to be used for performance analysis.  

ChatGPT and Random will execute from start to finish when the above command is used. For chatGPT you'll need your API key saved in a file called `secrets.txt` just as a single line, from your OpenAI account. Steps 1 & 2 are slow for the other methods and are skipped when running the CARE package with the above command.

Refer to each model below for details on their specific implementation from earlier steps:

### Similarity Baseline
Reaction representations used in the Similarity Baseline are found in `Similarity`.

2. Extract fingerprints using DRFP in `task2_baselines/get_drfp.ipynb`
3. Run `Similarity/example.sh` to perform a similarity search, for example:
```
python downstream_retrieval.py --pretrained_folder=Similarity/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction
```
The outputs will similarly be saved under retrieval_results.

### CREEP
Contrastive Reaction-EnzymE Pretraining (CREEP)
Outputs from model training and inference in our study are found in `CREEP`.

All of the terminal commands needed to run the scripts are provided in `CREEP/example.sh`, but an example is also provided here for convenience.

1. Go to the folder `task2_baselines/CREEP/`. Run finetuning training with default parameters:
```
python step_01_train_CREEP.py --output_model_dir=output/easy_split --train_split=easy_reaction_train
```

If you are running step 1 (training), you must have pretained ProtT5, SciBERT, and rxnfp models downloaded from [CARE_pretrained.zip](https://zenodo.org/records/12207966). Note that our batch size of 16 is optimized for a single 80GB GPU. Training for 40 epochs took about 36 hrs on a single H100 GPU. Training outputs will be saved in the `CREEP/output` directory. Various training parameters can be tuned using the argparser.

2. For extracting the reference protein representations and their cluster centers: 
```
python step_02_extract_CREEP.py --pretrained_folder=output/easy_split --dataset=all_proteins --modality=protein --get_cluster_centers
```

Note that this will take 0.5-1 hours on a single H100 GPU. If are manually starting from step 2 using CREEP, pretrained models can be downloaded from [CARE_pretrained.zip](https://zenodo.org/records/12207966).

For extracting the query reaction representations for each test set: 
```
python step_02_extract_CREEP.py --pretrained_folder=easy_split --dataset=easy_reaction_test --modality=reaction
python step_02_extract_CREEP.py --pretrained_folder=easy_split --dataset=easy_reaction_test --modality=text
```
Representations will be svaed in the output directory under `representations`.

3. Go back a folder and run the retrieval similarity search:
```
python downstream_retrieval.py --pretrained_folder=CREEP/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
python downstream_retrieval.py --pretrained_folder=CREEP/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=text --reference_modality=protein
```
The outputs will similarly be saved under `retrieval_results`.

### CLIPZyme
Outputs from model inference in our study are found in `CLIPZyme`. Running CLIPZyme requires installing the [CLIPZyme package](https://github.com/pgmikhael/clipzyme).

 1. Currently retraining is not availalbe, but will be added soon.
 2. First process protein sequences and reactions into the correct format using `CLIPZyme/step01_preparation.ipynb`. Then retrieve structures from the AF database and extract and process the representations of proteins and reactions using `CLIPZyme/step02_extraction.ipynb`.
 3. Go back a folder and run the retrieval similarity search (commands also provided in `ClIPZyme/example.sh` :
```
python downstream_retrieval.py --pretrained_folder=CLIPZyme/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
```

The outputs will similarly be saved under `retrieval_results`.
