cd ..

for split in easy medium hard
do
OUTPUT_DIR=output/"$split"_split
TEST_SET="$split"_reaction_test 

python downstream_retrieval.py --pretrained_folder=CLIPZyme/$OUTPUT_DIR --query_dataset=$TEST_SET --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein

done

