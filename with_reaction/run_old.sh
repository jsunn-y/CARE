export OUTPUT_DIR=../../output/ProteinDT/2modalities

CUDA_VISIBLE_DEVICES=1 python pretrain_step_01_GCLAP.py --output_model_dir="$OUTPUT_DIR" --dataset=CLEAN_split100.csv

CUDA_VISIBLE_DEVICES=1 python pretrain_step_01_GCLAP.py --output_model_dir="$OUTPUT_DIR" --dataset=CLEAN_train --mine_batch --loop_over_unique_EC

CUDA_VISIBLE_DEVICES=1 python pretrain_step_01_CREEP.py --output_model_dir="$OUTPUT_DIR" --dataset=CLEAN_train  --use_two_modalities

CUDA_VISIBLE_DEVICES=0 python pretrain_step_01_GSCLAP.py --dataset=SwissProtEnzymeCLAP/processed_data/trembl_smiles_240323_subsample500.csv --output_model_dir="$OUTPUT_DIR" --batch_size=6

#extracting the representations
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=easy_test_1360
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=medium_test_1291
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=hard_test_2257
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=reference_test_4908

CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=reference_test_3695

CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=testA_3668
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=testB_1591
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=testC_2964
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_CREEP.py --pretrained_folder="$OUTPUT_DIR" --dataset=all_proteins_CLEAN100

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testA_3668 --reference_dataset=testA_3668  --query_modality=reaction --reference_modality=protein -k=1

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testC_2964 --reference_dataset=testA_3668  --query_modality=reaction --reference_modality=protein -k=10

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testC_2964 --reference_dataset=testA_3668  --query_modality=reaction --reference_modality=protein -k=10 --use_cluster_center

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testB_1591 --reference_dataset=testA_3668  --query_modality=protein --reference_modality=reaction -k=1 --use_cluster_center

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=price_test_125 --reference_dataset=testA_3668  --query_modality=protein --reference_modality=reaction -k=1 --use_cluster_center

# python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testC_2964 --reference_dataset=testA_3668  --query_modality=text --reference_modality=protein -k=10 --use_cluster_center

python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testB_1591 --reference_dataset=testA_3668  --query_modality=protein --reference_modality=reaction -k=1 --use_cluster_center

python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=testC_2964 --reference_dataset=testA_3668  --query_modality=reaction --reference_modality=protein -k=1 --use_cluster_center

python downstream_exact_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=price_test_125 --reference_dataset=testA_3668  --query_modality=protein --reference_modality=reaction -k=1 --use_cluster_center


#for EC classification
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=price_test_125
CUDA_VISIBLE_DEVICES=1 python pretrain_step_02_extract_GCLAP.py --pretrained_folder="$OUTPUT_DIR" --dataset=new_test_295
python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=price_test_125 --reference_dataset=reference_test_4908  --query_modality=protein --reference_modality=text
python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=new_test_295 --reference_dataset=reference_test_4908  --query_modality=protein --reference_modality=text

python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=easy_test_1360 --reference_dataset=reference_test_4908  --query_modality=protein --reference_modality=text

python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=medium_test_1291 --reference_dataset=reference_test_4908  --query_modality=protein --reference_modality=text

#for other retrieval tasks
python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=hard_test_2257 --reference_dataset=reference_test_4908  --query_modality=text --reference_modality=protein

python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=easy_test_1360 --reference_dataset=reference_test_4908  --query_modality=text --reference_modality=protein

python downstream_faiss_retrieval.py --pretrained_folder="$OUTPUT_DIR" --query_dataset=reference_test_3695 --reference_dataset=reference_test_3695  --query_modality=reaction --reference_modality=protein


/disk1/jyang4/repos/BLAST/diamond blastp -d /disk1/jyang4/repos/BLAST/swissprot -q results_test.fasta -o results_test.blastp -k 1