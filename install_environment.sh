#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=score_hmr

conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME

# install pytorch using pip, update with appropriate cuda drivers if necessary
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install PHALP
pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

# install remaining requirements
pip install -r requirements.txt

# install source
pip install -e .
