#!/bin/sh

### RefCOCO/unc
python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 10 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/;

#### RefCOCO+/unc+
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc+/;

#### RefCOCO gref
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 10 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/gref/;

#### RefCOCO/gref_umd
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 10 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/gref_umd/;

#### ReferItGame/referit
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 10 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "Which region does the text {pseudo_query} describe?" --output_dir ./outputs/referit/;

#### Flickr
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler exponential --lr_exponential 0.8 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset flickr --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/flickr/;
