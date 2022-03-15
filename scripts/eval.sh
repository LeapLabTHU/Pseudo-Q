#!/bin/sh

### RefCOCO/unc
python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set testA --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/testA/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set testB --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/testB/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set val --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/val/;

### RefCOCO+/unc+
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc+_best_checkpoint.pth --eval_set testA --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc+/testA/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc+_best_checkpoint.pth --eval_set testB --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc+/testB/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc+_best_checkpoint.pth --eval_set val --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc+/val/;

### RefCOCOg gref
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/gref_best_checkpoint.pth --eval_set val --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/gref/val/;

### RefCOCOg gref_umd
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/gref_umd_best_checkpoint.pth --eval_set test --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/gref_umd/test/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/gref_umd_best_checkpoint.pth --eval_set val --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/gref_umd/val/;

### ReferItGame
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/referit_best_checkpoint.pth --eval_set test --prompt "Which region does the text {pseudo_query} describe?" --output_dir ./outputs/referit/test/;

#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/referit_best_checkpoint.pth --eval_set val --prompt "Which region does the text {pseudo_query} describe?" --output_dir ./outputs/referit/val/;

### Flickr30k
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset flickr --max_query_len 20 --data_root /root/project/data/referit_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/flickr_best_checkpoint.pth --eval_set test --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/flickr/test/;

python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset flickr --max_query_len 20 --data_root /root/project/data/referit_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/flickr_best_checkpoint.pth --eval_set val --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/flickr/val/;