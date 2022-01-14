# xview3-kohei-solution

## Usage

This repository assumes that the given data set is stored in the following locations:

```
$ ls data/input/xview3/*.csv
data/input/xview3/train.csv  data/input/xview3/validation.csv

$ ls data/input/xview3/downloaded
00a035722196ee86t.tar.gz  4455faa0cb4824f4t.tar.gz  85fe34d1aee53a7ft.tar.gz  c07f6ec980c2c149t.tar.gz
014261f774287442t.tar.gz  4518c556b38a5fa4t.tar.gz  864390795b0439b1t.tar.gz  c0831dbd6d7f3c56t.tar.gz
(snip)
```

### Training

#### Setup

```
# Setup: Create the virtual env
$ poetry ocnfig virtualenvs.in-project true
$ poetry install
$ poetry run pip install albumentations timm segmentation_models_pytorch torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Training

* To save storage space, my preprocessing code assumes that the input image file is the original file in .tar.gz format. It does NOT assume pre-extracted files.
* The batch size and other settings are based on the assumption that two RTX3080 GPU cards are used.

    ```
$ bash train.sh
```

### Inference

My containerized inference code follows the xView3 evaluation protocol. The detailed usage of the xview3 evaluation protocol is described in https://iuu.xview.us/verify.

```
## Setup: Build the docker image -----
$ docker build --no-cache -t kohei-xview3 .

## Inference -----
$ docker run \
    --shm-size 16G \
    --gpus=1 \
    --mount type=bind,source=/home/xv3data,target=/on-docker/xv3data \
    kohei-xview3 \
    /on-docker/xv3data/ \
    0157baf3866b2cf9v \
    /on-docker/xv3data/prediction/prediction.csv
```
