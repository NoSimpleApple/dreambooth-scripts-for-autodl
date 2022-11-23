#!/bin/bash

# Download the model
# https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animefull-pruned.tar
# https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt
# tar -cf animefull-pruned.tar

# do not change this if you know about your risk
export AUTODL_TMP="/root/autodl-tmp"

# path of baseline model and external vae file
export BASE_MODEL_PATH="${AUTODL_TMP}/models"
export EX_VAE_PATH="${BASE_MODEL_PATH}/vae"

# literally and used by the dir name containing converted model
export BASE_MODEL_NAME="mmk-e_fucked"

# path to save checkpoint
export OUTPUT_PATH="${AUTODL_TMP}/bussin"

# path contains your stunning creation
export DRUG_PATH="${AUTODL_TMP}/drugs"
export DRUG_FILENAME_PREFIX="drug-"

# convert model needed
export DUMP_MODEL_PATH="${AUTODL_TMP}/${BASE_MODEL_NAME}"
export BASE_MODEL_CONF_PATH="${BASE_MODEL_PATH}/config.yaml"

################
# training
################
# anaconda venv name
export CONDA_VENV_NAME="aichemy"
# venv path
export EX_VENV_PATH=""

# this is relative from your current working dir
export TRAIN_CONFIG="base_config.yaml"

export TO_EPOCH=50
# will override $TO_EPOCH
export TO_STEPS=3500
