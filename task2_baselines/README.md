## Baselines for task 2 (reaction to EC/protein retrieval)
The results of EC retrieval for each method and each split is found in `results_summary` as .csv files, for use in downstream performance analysis. 

### Random
```
python rank_tabulate_random.py
```

### ChatGPT
For this, run the ChatGPT notebook here and follow the instructions to insert your API key: `ChatGPT/ChatGPT.ipynb`.
This contains two cells, one for reaction only, and one for reaction and text.

## Other methods:
For the other benchmarks, results can be reproduced at a high level  by following these steps: 

1. Model trainining for CREEP and CLIPZyme. Similarity baseline can skip to step 2. Pretrained models are available on [huggingface](https://huggingface.co/jsunn-y/CARE_pretrained).
2. Extract representations in the format of .npy files containing arrays of representations from multiple modalities (such as protein, reaction, and text).
3. A similarity search between the train and test set is performed using `downstream_retrieval.py`. Retrieval similarites are outputed as .npy arrays under the respective method and model folder in `retrieval_results`.
4. The retrieval similarities are processed to obtain a ranking of EC numbers using `rank_tabulate_similarities.py`. The outputs will be .csv files saved to their respective folders in `results_summary`, to be used for performance analysis.  
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

If you are running step 1 (training), you must have pretained ProtT5, SciBERT, and rxnfp models, which will be pulled from huggingface hub. Note that our batch size of 16 is optimized for a single 80GB GPU. Training for 40 epochs took about 36 hrs on a single H100 GPU. Training outputs will be saved in the `CREEP/output` directory. Various training parameters can be tuned using the argparser.

2. For extracting the reference protein representations and their cluster centers: 
```
python step_02_extract_CREEP.py --pretrained_folder=output/easy_split --dataset=all_proteins --modality=protein --get_cluster_centers
```

Note that this will take 0.5-1 hours on a single H100 GPU. If are manually starting from step 2 using CREEP, pretrained models can be downloaded from [huggingface](https://huggingface.co/jsunn-y/CARE_pretrained).

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

 1. Follow the processing, retraining, and inference steps provied in `CLIPZyme/CARE_for_CLIPZyme.ipynb`. Alternatively, our pretrained CLIPZyme models can be downloaded from [huggingface](https://huggingface.co/jsunn-y/CARE_pretrained).
 2. Go back a folder and run the retrieval similarity search (commands also provided in `ClIPZyme/example.sh` :
```
python downstream_retrieval.py --pretrained_folder=CLIPZyme/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
```

The outputs will similarly be saved under `retrieval_results`.
