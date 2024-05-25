cd ..
python downstream_retrieval.py --pretrained_folder=Similarity/output/easy_split --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction

python downstream_retrieval.py --pretrained_folder=Similarity/output/medium_split --query_dataset=medium_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction

python downstream_retrieval.py --pretrained_folder=Similarity/output/hard_split --query_dataset=hard_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction

