for split in easy medium hard
do
OUTPUT_DIR=output/"$split"_split
TRAIN_SET="$split"_reaction_train
TEST_SET="$split"_reaction_test 

cd CREEP
python step_02_extract_CREEP.py --pretrained_folder=$OUTPUT_DIR --dataset=all_proteins --modality=protein --get_cluster_centers

python step_02_extract_CREEP.py --pretrained_folder=$OUTPUT_DIR --dataset=$TEST_SET --modality=reaction
python step_02_extract_CREEP.py --pretrained_folder=$OUTPUT_DIR --dataset=$TEST_SET --modality=text

# python step_02_extract_CREEP.py --pretrained_folder=$OUTPUT_DIR --dataset=$TRAIN_SET --modality=reaction --get_cluster_centers

cd ..

python downstream_retrieval.py --pretrained_folder=CREEP/$OUTPUT_DIR --query_dataset=$TEST_SET --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein

python downstream_retrieval.py --pretrained_folder=CREEP/$OUTPUT_DIR --query_dataset=$TEST_SET --reference_dataset=all_ECs --query_modality=text --reference_modality=protein

#similarity baseline (do this separately)
#python downstream_retrieval.py --pretrained_folder=Similarity/#OUTPUT_DIR --query_dataset=$TEST_SET --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction

done

