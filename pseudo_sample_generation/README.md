# Generating Pseudo-samples

## Usage

### Detection Results Preparation
First, we adopt the [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome) to get the object proposals of unlabeled images. Second, we utilize the [bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch) to get attributes of objects.

- You can download the detection results from [Google Drive](https://drive.google.com/file/d/1rC2TEnLlQe-URGRh5_rpmGGExvQDFcZV/view?usp=sharing) and put them in the right place.

    ```
    $ mv detection_results.tar.gz ./Pseudo-Q/data/
    $ tar -zxvf detection_results.tar.gz
    ```

- Or you can follow the tutorial in the above two repositories to get detection results by your own. Here, I provide [a code script](get_detection_results.py) to help researchers to generate object detection and attribute recognition results based on [bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch). 
    
    1. Please follow the instruction in bottom-up-attention to build the environment.
    1. Before running the code, you should get the [config file](extract-bua-caffe-r152.yaml), download the checkpoint weights from [Google Drive](https://drive.google.com/file/d/10oU4Zr06YOX7PlgJ8rDpNjXwXgupNNUL/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/7410b68588b342c481f6/), and put these files on the right path.
    1. Run the code with following command on the bottom-up-attention codebase.


    ```
    $ OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python get_detection_results.py --load_dir data/pretrained_model --image_dir /home/data/referit_data/other/images/mscoco/images/train2014 --out_path /cluster/home1/hjjiang/CVPR2022/pseudo_dataset_after_debug_1027/unc/r152_attr_detection_results/ --image_list_file /cluster/home1/hjjiang/CVPR2022/Faster-R-CNN-with-model-pretrained-on-Visual-Genome_2080/statistic/unc/unc_train_imagelist_split0.txt --vg_dataset unc --cuda --split_ind 0
    ```

### Generation of Pseudo-samples
Please download the images of RefCOCO or other dataset, to ```../data/image_data``` first.

```
$ cd pseudo_sample_generation
$ bash scripts/generate_pseudo_data_unc.sh
```

The pseudo samples will be stored in ```../data/pseudo_sample/```.



