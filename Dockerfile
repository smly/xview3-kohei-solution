FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

SHELL [ "/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -yq software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -yq apt-transport-https \
    gcc \
    g++ \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libstdc++6 \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /home/xview3/
WORKDIR /home/xview3/
ENV HOME=/home/xview3

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

RUN source $HOME/.poetry/env && poetry install
RUN source $HOME/.poetry/env && poetry run pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN source $HOME/.poetry/env && poetry run pip install opencv-python
RUN source $HOME/.poetry/env && poetry run pip install albumentations timm

COPY v13/ep59.pth $HOME/v13ep59.pth
COPY v16/ep19.pth $HOME/v16ep19.pth
COPY v77/*.pth $HOME/
COPY run_inference.sh $HOME
COPY main.py $HOME
COPY models_hrnet.py $HOME

RUN apt-get update \
    && apt-get install -yq cmake libboost-all-dev shapelib libshp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN source $HOME/.poetry/env && poetry run pip install pyproj dask xarray
RUN source $HOME/.poetry/env && poetry run pip install "pybind11[global]"
RUN source $HOME/.poetry/env && git clone https://github.com/fbriol/gshhg.git gshhg_ \
    && cd gshhg_ \
    && source $HOME/.poetry/env \
    && poetry run python setup.py build \
    && poetry run python setup.py install

COPY GSHHS_shp $HOME/GSHHS_shp

ENTRYPOINT [ "/home/xview3/run_inference.sh" ]
