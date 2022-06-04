#!/usr/bin/env bash

OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 0 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 1 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 2 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 3 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 4 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 5 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 6 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 7 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 8 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 9 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 10 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 11 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 12 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 13 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 14 --topn 3 --each_image_query 6;
OMP_NUM_THREADS=4 python pseudo_query_generation.py --vg_dataset_path ../data/image_data --vg_dataset unc --split_ind 15 --topn 3 --each_image_query 6;
python ./utils/merge_file.py ../data/pseudo_samples/unc/top3_query6/ unc;
python ./utils/post_process.py ../data/pseudo_samples/unc/top3_query6/unc/ unc;

