## Baselines for task 2 (reaction to EC/protein retrieval)
The results of EC retrieval for each method and each split is found in `results_summary` as .csv files, for use in downstream performance analysis. 

Alternatively, these results can be reproduced at a high level by following these steps: 


1. Model trainining for CREEP and CLIPZyme. Similarity baseline can skip to step 2.
2. Extract representations in the format of .npy files containing arrays of representations from multiple modalities (such as protein, reaction, and text).
3. A similarity search between the train and test set is performed using `downstream_retrieval.py`. Retrieval similarites are outputed as .npy arrays under the respective method and model folder in `retrieval_results`.
4. The retrieval similarities are processed to obtain a ranking of EC numbers using `tabulate_results.ipynb`. The outputs will be .csv files saved to their respective folders in `results_summary`, to be used for performance analysis.  

Refer to each model below for details on their specific implementation:

### Similarity Baseline
2. Extract fingerprints using DRFP in `task2_baselines/get_drfp.ipynb`
3. Run `Similarity/example.sh` to perform a similarity search, for example:
```
python downstream_retrieval.py --pretrained_folder=Similarity/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction
```

### CREEP
Contrastive Reaction-EnzymE Pretraining (CREEP)

All of the terminal commands needed to run the scripts are provided in `CREEP/example.sh`, but an example is also provided here for convenience.

1. Go to the folder `task2_baselines/CREEP/`. Run finetuning training with default parameters:
```
python step_01_train_CREEP.py --output_model_dir=output/easy_split --train_split=easy_reaction_train
```

Note that our batch size of 16 is optimized for a single 80GB GPU. Training for 40 epochs took about 36 hrs on a single H100 GPU. Training outputs will be saved in the `CREEP/output` directory. Various training parameters can be tuned using the argparser.

2. For extracting the reference protein representations and their cluster centers: 
```
python step_02_extract_CREEP.py --pretrained_folder=output/easy_split --dataset=all_proteins --modality=protein --get_cluster_centers
```

Note that this will take 0.5-1 hours on a single H100 GPU.

For extracting the query reaction representations for each test set: 
```
python step_02_extract_CREEP.py --pretrained_folder=easy_split --dataset=easy_reaction_test --modality=reaction
python step_02_extract_CREEP.py --pretrained_folder=easy_split --dataset=easy_reaction_test --modality=text
```
Representations will be svaed in the output directory under `representations`.

3. Run the retrieval similarity search:
Go back a folder (retrieval similarity search can be run on any representations that are formatted in the same way as those from CREEP step 2.
```
python downstream_retrieval.py --pretrained_folder=CREEP/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
python downstream_retrieval.py --pretrained_folder=CREEP/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=text --reference_modality=protein
```
The outputs will be saved under `retrieval_results`.

### CLIPZyme
First process the data and download the necessary protein structures from the AF database. Optionally retrain the model using sequences clustered at 50% sequence identity.

Extract and process the representations of reaction and protein modalities into the same format as CREEP using `inference.ipynb`.

Then run the retrieval similarity search form `task2_baselines`
```
python downstream_retrieval.py --pretrained_folder=CLIPZyme/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein
```
The outputs will similarly be saved under `retrieval_results` and can be further analyzed and visualized in `retrieval_analysis_metrics.ipynb`.

### Chemcrow

```pip install paperscraper```

Response:

```
>>> chem_model.run("CLassify the EC of this protein sequence: MAFASKFARSKTILSFLRPCRQLHSTPKSTGDVTVLSPVKGRRRLPTCWSSSLFPLAIAASATSFAYLNLSNPSISESSSALDSRDITVGGKDSTEAVVKGEYKQVPKELISQLKTILEDNLTTDYDERYFHGKPQNSFHKAVNIPDVVVFPRSEEEVSKILKSCNEYKVPIVPYGGATSIEGHTLAPKGGVCIDMSLMKRVKALHVEDMDVIVEPGIGWLELNEYLEEYGLFFPLDPGPGASIGGMCATRCSGSLAVRYGTMRDNVISLKVVLPNGDVVKTASRARKSAAGYDLTRLIIGSEGTLGVITEITLRLQKIPQHSVVAVCNFPTVKDAADVAIATMMSGIQVSRVELLDEVQIRAINMANGKNLTEAPTLMFEFIGTEAYTREQTQIVQQIASKHNGSDFMFAEEPEAKKELWKIRKEALWACYAMAPGHEAMITDVCVPLSHLAELISRSKKELDASSLLCTVIAHAGDGNFHTCIMFDPSSEEQRREAERLNHFMVHSALSMDGTCTGEHGVGTGKMKYLEKELGIEALQTMKRIKKTLDPNDIMNPGKLIPPHVCF")


> Entering new RetryAgentExecutor chain...
DEBUG:openai:message='Request to OpenAI API' method=post path=https://api.openai.com/v1/chat/completions
DEBUG:openai:api_version=None data='{"messages": [{"role": "system", "content": "\\nYou are an AI system.\\n\\n\\n\\nYou can only respond with a single complete\\n\\"Thought, Action, Action Input\\" format\\nOR a single \\"Final Answer\\" format.\\n\\nComplete format:\\n\\nThought: (reflect on your progress and decide what to do next)\\nAction: (the action name, should be one of [Python_REPL, Wikipedia, Name2SMILES, Mol2CAS, SMILES2Name, PatentCheck, MolSimilarity, SMILES2Weight, FunctionalGroups, ExplosiveCheck, ControlChemCheck, SimilarityToControlChem, SafetySummary, LiteratureSearch])\\nAction Input: (the input string to the action)\\n\\nOR\\n\\nFinal Answer: (the final answer to the original input question)\\n"}, {"role": "user", "content": "\\nAnswer the question below using the following tools:\\n\\n    Python_REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.\\n    Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.\\n    Name2SMILES: Input a molecule name, returns SMILES.\\n    Mol2CAS: Input molecule (name or SMILES), returns CAS number.\\n    SMILES2Name: Input SMILES, returns molecule name.\\n    PatentCheck: Input SMILES, returns if molecule is patented. You may also input several SMILES, separated by a period.\\n    MolSimilarity: Input two molecule SMILES (separated by \'.\'), returns Tanimoto similarity.\\n    SMILES2Weight: Input SMILES, returns molecular weight.\\n    FunctionalGroups: Input SMILES, return list of functional groups in the molecule.\\n    ExplosiveCheck: Input CAS number, returns if molecule is explosive.\\n    ControlChemCheck: Input CAS number, True if molecule is a controlled chemical.\\n    SimilarityToControlChem: Input SMILES, returns similarity to controlled chemicals.\\n    SafetySummary: Input CAS number, returns a summary of safety information.The summary includes Operator safety, GHS information, Environmental risks, and Societal impact.\\n    LiteratureSearch: Useful to answer questions that require technical knowledge. Ask a specific question.\\n\\nUse the tools provided, using the most specific tool available for each action.\\nYour final answer should contain all information necessary to answer the question and subquestions.\\n\\nIMPORTANT: Your first step is to check the following, in this order, and plan your steps accordingly:\\n1. Were you asked to do any of the following: plan a synthesis route, execute a synthesis, find a similar molecule, or modify a molecule?\\nIf so, your first step is to check if the molecule is a controlled chemical. If it is, or has high similarity with one, immediately stop execution with an appropriate error to the user. Do not continue.\\n2. Does the question involve any molecules? If so, as a first step, check if any are controlled chemicals. If any are, include a warning in your final answer.\\n3. Were you asked to plan a synthesis route? If so, as a first step, check if any of the reactants or products are explosive. If any are, include a warning in your final answer.\\n4. Were you asked to execute a synthesis route? If so, check if any of the reactants or products are explosive. If any are, ask the user for permission to continue.\\nDo not skip these steps.\\n\\n\\nQuestion: CLassify the EC of this protein sequence: MAFASKFARSKTILSFLRPCRQLHSTPKSTGDVTVLSPVKGRRRLPTCWSSSLFPLAIAASATSFAYLNLSNPSISESSSALDSRDITVGGKDSTEAVVKGEYKQVPKELISQLKTILEDNLTTDYDERYFHGKPQNSFHKAVNIPDVVVFPRSEEEVSKILKSCNEYKVPIVPYGGATSIEGHTLAPKGGVCIDMSLMKRVKALHVEDMDVIVEPGIGWLELNEYLEEYGLFFPLDPGPGASIGGMCATRCSGSLAVRYGTMRDNVISLKVVLPNGDVVKTASRARKSAAGYDLTRLIIGSEGTLGVITEITLRLQKIPQHSVVAVCNFPTVKDAADVAIATMMSGIQVSRVELLDEVQIRAINMANGKNLTEAPTLMFEFIGTEAYTREQTQIVQQIASKHNGSDFMFAEEPEAKKELWKIRKEALWACYAMAPGHEAMITDVCVPLSHLAELISRSKKELDASSLLCTVIAHAGDGNFHTCIMFDPSSEEQRREAERLNHFMVHSALSMDGTCTGEHGVGTGKMKYLEKELGIEALQTMKRIKKTLDPNDIMNPGKLIPPHVCF\\n"}, {"role": "assistant", "content": "\\nThought: \\n"}], "model": "gpt-4-0613", "max_tokens": null, "stream": false, "n": 1, "temperature": 0.1, "stop": ["\\nObservation:", "\\n\\tObservation:"]}' message='Post details'
DEBUG:openai:message='OpenAI API response' path=https://api.openai.com/v1/chat/completions processing_ms=5789 request_id=req_d913fd30ab5940593c879aad621cc055 response_code=200
Thought: The question asks for the EC (Enzyme Commission) classification of a given protein sequence. This is a bioinformatics task that requires the use of a tool that can analyze protein sequences and predict their function. However, none of the provided tools are capable of performing this task. Therefore, I am unable to provide an answer to this question.
Final Answer: I'm sorry, but I am unable to classify the EC of the given protein sequence using the tools provided.

> Finished chain.

"I'm sorry, but I am unable to classify the EC of the given protein sequence using the tools provided."
```