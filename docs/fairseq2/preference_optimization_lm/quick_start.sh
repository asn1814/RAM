#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################################################################
# File Name   : quick_start.sh
# Description : Run this script with './quick_start.sh' and your fairseq2 
#               environment active on a node with sufficient GPUs
#################################################################################

# get fairseq2_ram/docs/preference_optimization_lm
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# get data
echo Begin loading data
CARD_DIR=${SCRIPT_DIR}/datasets/cards
DATA_DIR=${SCRIPT_DIR}/datasets/oasst2_data
DATA_PREP_PY=${SCRIPT_DIR}/datasets/openassistant2_llama3_data_prep.py

python ${DATA_PREP_PY} --data_dir ${DATA_DIR} --card_dir ${CARD_DIR}
echo Data loading finished!

# train
echo Begin training
export FAIRSEQ2_ASSET_DIR=${CARD_DIR}
CONFIG_YAML=${SCRIPT_DIR}/example_configurations/example_dpo_config.yaml
OUTPUT_DIR=${SCRIPT_DIR}/quick_start_dpo_training_output/

torchrun --standalone --nproc-per-node 8 --no-python fairseq2 lm preference_finetune ${OUTPUT_DIR} --config-file ${CONFIG_YAML} --config dataset=openassistant2_preference_llama3_train --no-sweep-dir
echo Training finished!

# convert model to huggingface
echo Begin conversion to HuggingFace format
FS2_TO_HF_PY=${SCRIPT_DIR}/../../ckpt_convert_fairseq2_hf.py
CHECKPOINT_PATH=${OUTPUT_DIR}/checkpoints/last_step
SAVE_PATH=${OUTPUT_DIR}/huggingface_saves/last_step
ARCH=llama3_8b

python ${FS2_TO_HF_PY} --fairseq2_train_ckpt_dir ${CHECKPOINT_PATH} --save_path ${SAVE_PATH} --arch ${ARCH} 
echo Conversion to HuggingFace format finished!

# see metrics with tensorboard
echo Start Tensorboard
tensorboard --logdir ${OUTPUT_DIR}/tb
