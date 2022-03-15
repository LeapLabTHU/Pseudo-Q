# Pseudo-Q
<p align="center"> <img src='docs/framework.jpg' align="center" height="200px"> </p>

This repository is the official Pytorch implementation for CVPR2022 paper **Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding**.

**Update on 2022/03/15: Release the training code.**

## Reference

If you find our project useful in your research, please consider citing:

```
@inproceedings{jiang2022pseudoq,
  title={Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding},
  author={Jiang, Haojun and Lin, Yuanze and Han, Dongchen and Song, Shiji and Huang, Gao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgments](#acknowledgments)

## Introduction
We present a novel method, named **Pseudo-Q**, to automatically generate pseudo language queries for supervised training. Our method leverages an off-the-shelf object detector to identify visual objects from unlabeled images, and then language queries for these objects are obtained in an unsupervised fashion with a pseudo-query generation module. Extensive experimental results demonstrate that our method has two notable benefits: (1) it can reduce human annotation costs significantly, e.g., 31% on RefCOCO without degrading original model's performance under the fully supervised setting, and (2) without bells and whistles, it achieves superior or comparable performance compared to state-of-the-art weakly-supervised visual grounding methods on all the five datasets we have experimented. For more details. please refer to our paper.

## Usage

### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- [Pytorch-Bert 0.6.2](https://pypi.org/project/pytorch-pretrained-bert/)
- Check [requirements.txt](requirements.txt) for other dependencies. 


### Data Preparation
1.You can download the images from the original source and place them in ./data/image_data folder:
- [RefCOCO and ReferItGame](https://github.com/lichengunc/refer)
- [Flickr30K Entities](https://bryanplummer.com/Flickr30kEntities/)

2.The generated pseudo region-query pairs can be download from [Tsinghua Cloud]().
```
mkdir data
mv pseudo_samples.tar.gz ./data/
tar -zxvf pseudo_samples.tar.gz
```

<!--3.For your convenience, we also provide the data(RefCOCO and ReferItGame) from [Tsinghua Cloud](). Since we do not have the copyright of the images in Flickr30K Entities dataset, you have to download from the above link.
    
```
mv image_data.tar.gz ./data/
tar -zxvf image_data.tar.gz
```
-->


### Pretrained Checkpoints
1.You can download the DETR checkpoints from [Tsinghua Cloud](). These checkpoints should be downloaded and move to the [checkpoints](./checkpoints) directory.

```
mkdir checkpoints
mv detr_checkpoints.tar.gz ./checkpoints/
tar -zxvf checkpoints.tar.gz
```

2.Checkpoints that trained on our pseudo-samples can be downloaded from [Tsinghua Cloud](). You can evaluate the checkpoints following the instruction right below.

```
mv pseudoq_checkpoints.tar.gz ./checkpoints/
tar -zxvf pseudoq_checkpoints.tar.gz
```

### Training and Evaluation

1.  Training on RefCOCO. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 10 --batch_size 32 --lr 0.00025 --lr_bert 0.000025 --lr_visu_cnn 0.000025 --lr_visu_tra 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/;
    ```

    Please refer to [scripts/train.sh](scripts/train.sh) for training commands on other datasets.

2.  Evaluation on RefCOCO.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/pseudo_samples/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set testA --prompt "find the region that corresponds to the description {pseudo_query}" --output_dir ./outputs/unc/testA/;
    ```

## Contacts
jhj20 at mails dot tsinghua dot edu dot cn

Any discussions or concerns are welcomed!


## Acknowledge
This codebase is based on [TransVG](https://github.com/djiajunustc/TransVG).
