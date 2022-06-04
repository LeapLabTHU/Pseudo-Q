# Generating Pseudo-samples

## Usage

### Detection Results Preparation
First, we adopt the [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome) to get the object proposals of unlabeled images. Second, we utilize the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to get attributes of objects.

- You can download the detection results from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/7ea0adc0c4d544519191/?dl=1) and put them in the right place.

    ```
    $ mv detection_results.tar.gz ./Pseudo-Q/data/
    $ tar -zxvf detection_results.tar.gz
    ```

- Or you can follow the tutorial in the above two repositories to get detection results by your own.



### Generation of Pseudo-samples
Please download the images of RefCOCO or other dataset, to ```../data/image_data``` first.

```
$ cd pseudo_sample_generation
$ bash scripts/generate_pseudo_data_unc.sh
```

The pseudo samples will be stored in ```../data/pseudo_sample/```.



