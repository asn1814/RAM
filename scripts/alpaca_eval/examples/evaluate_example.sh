#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################################################################
#     File Name           :     evaluate_example.sh
#     Description         :     Run this script with 'sbatch evaluate_example.sh' with the alpacaeval conda environment activated
#################################################################################
### SBATCH directives to specify job configuration
#SBATCH --job-name=ae2example
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/alpacaeval2example-%j.out
#SBATCH --error=/checkpoint/%u/jobs/alpacaeval2example-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --tasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --constraint=volta32gb
#SBATCH --partition=dummy_partition
#SBATCH --time 01:00:00

### setup to get the reference and model outputs
HF_MODEL_PATH= # filepath to Llama 3 8B Instruct in HuggingFace format here
MODEL_NAME=llama_3_8b_instruct_example
GET_OUTPUTS_PY=RAM/scripts/alpaca_eval/get_outputs.py
OUTPUTS_PATH=RAM/scripts/alpaca_eval/generations/${MODEL_NAME}_generations.json
DEFAULT_OUTPUTS_PATH=RAM/scripts/alpaca_eval/generations/default_generations.json

### get the reference outputs
python ${GET_OUTPUTS_PY} --default_reference True --outputs_file ${DEFAULT_OUTPUTS_PATH}
### get the model outputs
python ${GET_OUTPUTS_PY} --model_file ${HF_MODEL_PATH} --model_name ${MODEL_NAME} --outputs_file ${OUTPUTS_PATH} --template llama3_template

### annotate with Llama 3.1 70b Instruct
LEADERBOARD_PATH=RAM/scripts/alpaca_eval/examples/leaderboard.csv
ANNOTATORS_CONFIG_PATH=RAM/scripts/alpaca_eval/evaluator_configs/llama3.1_70b_instruct/configs.yaml
alpaca_eval make_leaderboard --leaderboard_path ${LEADERBOARD_PATH} --all_model_outputs ${OUTPUTS_PATH} --reference_outputs ${DEFAULT_OUTPUTS_PATH} --annotators_config ${ANNOTATORS_CONFIG_PATH}