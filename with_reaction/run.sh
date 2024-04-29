export OUTPUT_DIR=../../output/ProteinDT/240418

CUDA_VISIBLE_DEVICES=1 python pretrain_step_01_CREEP.py --output_model_dir="$OUTPUT_DIR" --dataset=train_240423
CUDA_VISIBLE_DEVICES=1 python pretrain_step_01_CREEP.py --output_model_dir="$OUTPUT_DIR" --dataset=train_240423  --use_two_modalities

#extracting the representations

CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_sets --dataset=all_proteins --modality=protein
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_sets --dataset=all_reactions --modality=reaction
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_sets --dataset=all_ECs --modality=text

for dataset in rxn2ec_easy_test rxn2ec_medium_test rxn2ec_hard_test
do
    CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_240423 --dataset=$dataset --modality=reaction
    CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_240423 --dataset=$dataset --modality=text
done

for dataset in protein2ec_easy_test protein2ec_hard_test protein2ec_price98_test
do
    python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_240423 --dataset=$dataset --modality=protein
    python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset_folder=test_240423 --dataset=$dataset --modality=text
done

# for dataset in rxn2ec_hard_test
# do
# python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein -k=10 --use_cluster_center
# done


for dataset in rxn2ec_easy_test rxn2ec_medium_test rxn2ec_hard_test
do
python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein -k=10 --use_cluster_center
python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=text --reference_modality=protein -k=10 --use_cluster_center
done

#similarity baseline (doesn't really make sense to do this because the test clusters are removed)
# for dataset in rxn2ec_hard_test
# do
# python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=reaction --reference_modality=reaction -k=10 --use_cluster_center
# done

for dataset in protein2ec_easy_test protein2ec_hard_test protein2ec_price98_test
do
python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=protein --reference_modality=reaction -k=10 --use_cluster_center
python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=protein --reference_modality=text -k=10 #don't use cluster center here
done

# for dataset in protein2ec_easy_test protein2ec_hard_test protein2ec_price98_test
# do
# python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=$dataset --reference_dataset=all_ECs --query_modality=protein --reference_modality=text -k=10
# done



/disk1/jyang4/repos/BLAST/diamond blastp -d /disk1/jyang4/repos/BLAST/swissprot -q results_test.fasta -o results_test.blastp -k 1