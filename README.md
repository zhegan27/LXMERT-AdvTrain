# Adversarial Training of LXMERT 
This repository contains the PyTorch code of [VILLA](https://arxiv.org/pdf/2006.06195.pdf) (NeurIPS 2020 Spotlight) that supports adversarial training (finetuning) of [LXMERT](https://arxiv.org/pdf/1908.07490.pdf) on [VQA](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), and [NLVR2](http://lil.nlp.cornell.edu/nlvr/). Most of the code in this repo are copied/modified from [LXMERT](https://github.com/airsplay/lxmert).

For details on UNITER adversarial pre-training and finetuning, please refer to the [main VILLA repo](https://github.com/zhegan27/VILLA).

![Overview of VILLA](villa_framework.png)

## Results
This repo can be used to reproduce the following results.

| Method           | VQA (test-dev) | VQA (test-std)  | GQA (test-dev) | GQA (test-std) | NLVR2 (dev)   | NLVR2 (test-P) |
:-----------:      |:-----------:   |:-----------:    |:-----------:   |:-----------:   |:-----------:  |:-----------:   |
| LXMERT           | 72.50% | 72.52%  | 59.92% | 60.28%  | 74.72%  | 74.75% |
| LXMERT-AdvTrain  | 73.02% | 73.18%  | 60.98% | 61.12%  | 75.98%  | 75.73% |


## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## VQA
*NOTE*: Please follow the official [LXMERT](https://github.com/airsplay/lxmert) repo to download the pre-trained checkpoint and get all the VQA features and processed data ready.

1. Organize the downloaded VQA data with the following folder structure:
    ```
    ├── finetune 
    ├── img_db
    │   ├── train2014_obj36.tsv
    │   └── val2014_obj36.tsv
    ├── pretrained
    │   └── model_LXRT.pth
    └── txt_db
        ├── minival.json
        ├── nominival.json
        └── train.json

    ```

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/src` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)


3. Run finetuning for the VQA task.
    ```bash
    # verify on a small training set
    bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
    bash run/vqa_adv_finetune.bash 0 vqa_lxr955_adv_tiny --tiny

    # standard finetuning
    bash run/vqa_finetune.bash 0 vqa_lxr955

    # adversarial finetuning
    bash run/vqa_adv_finetune.bash 0 vqa_lxr955_adv
    ```

4. Run inference for the VQA task and then evaluate.
    ```bash
    # local validation
    bash run/vqa_test.bash 0 vqa_lxr955_results --test minival --load /storage/vqa_lxr955/BEST
    bash run/vqa_test.bash 0 vqa_lxr955_adv_results --test minival --load /storage/vqa_lxr955_adv/BEST

    # submission to VQA test server
    # you need download the test data first
    bash run/vqa_test.bash 0 vqa_lxr955_results --test test --load /storage/vqa_lxr955/BEST
    bash run/vqa_test.bash 0 vqa_lxr955_adv_results --test test --load /storage/vqa_lxr955_adv/BEST
    ```

## GQA
*NOTE*: Please follow the official [LXMERT](https://github.com/airsplay/lxmert) repo to download the GQA processed data and features.

1. Organize the downloaded GQA data with the following folder structure:
    ```
    ├── finetune 
    ├── img_db
    │   ├── gqa_testdev_obj36.tsv
    │   └── vg_gqa_obj36.tsv
    ├── pretrained
    │   └── model_LXRT.pth
    └── txt_db
        ├── testdev.json
        ├── train.json
        └── valid.json

    ```

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```

3. Run finetuning for the GQA task.
    ```bash
    # verify on a small training set
    bash run/gqa_finetune.bash 0 gqa_lxr955_tiny --tiny
    bash run/gqa_adv_finetune.bash 0 gqa_lxr955_adv_tiny --tiny

    # standard finetuning
    bash run/gqa_finetune.bash 0 gqa_lxr955

    # adversarial finetuning
    bash run/gqa_adv_finetune.bash 0 gqa_lxr955_adv
    ```

4. Run inference for the GQA task and then evaluate.
    ```bash
    # local validation
    bash run/gqa_test.bash 0 gqa_lxr955_results --load /storage/gqa_lxr955/BEST --test testdev --batchSize 1024
    bash run/gqa_test.bash 0 gqa_lxr955_adv_results --load /storage/gqa_lxr955_adv/BEST --test testdev --batchSize 1024

    # submission to GQA test server
    # you need download the test data first
    bash run/gqa_test.bash 0 gqa_lxr955_results --load /storage/gqa_lxr955/BEST --test submit --batchSize 1024
    bash run/gqa_test.bash 0 gqa_lxr955_adv_results --load /storage/gqa_lxr955_adv/BEST --test submit --batchSize 1024
    ```

## NLVR2
*NOTE*: Please follow the official [LXMERT](https://github.com/airsplay/lxmert) repo to download the NLVR2 processed data and features.

1. Organize the downloaded NLVR2 data with the following folder structure:
    ```
    ├── finetune 
    ├── img_db
    │   ├── train_obj36.tsv
    │   └── valid_obj36.tsv
    ├── pretrained
    │   └── model_LXRT.pth
    └── txt_db
        ├── test.json
        ├── train.json
        └── valid.json

    ```

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```

3. Run finetuning for the NLVR2 task.
    ```bash
    # verify on a small training set
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955_tiny --tiny
    bash run/nlvr2_adv_finetune.bash 0 nlvr2_lxr955_adv_tiny --tiny

    # standard finetuning
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955

    # adversarial finetuning
    bash run/nlvr2_adv_finetune.bash 0 nlvr2_lxr955_adv
    ```

4. Run inference for the VQA task and then evaluate.
    ```bash
    # inference on public test split
    bash run/nlvr2_test.bash 0 nlvr2_lxr955_results --load /storage/nlvr2_lxr955/BEST --test test --batchSize 1024
    bash run/nlvr2_test.bash 0 nlvr2_lxr955_adv_results --load /storage/nlvr2_lxr955_adv/BEST --test test --batchSize 1024
    ```

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{gan2020large,
  title={Large-Scale Adversarial Training for Vision-and-Language Representation Learning},
  author={Gan, Zhe and Chen, Yen-Chun and Li, Linjie and Zhu, Chen and Cheng, Yu and Liu, Jingjing},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={EMNLP},
  year={2019}
}
```

## License

MIT
