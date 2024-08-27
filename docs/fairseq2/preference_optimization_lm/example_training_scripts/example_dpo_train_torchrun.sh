#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Setting environment variables for the job.
## OUTPUT_DIR specifies the location that the trained model and training metrics should be stored.
OUTPUT_DIR=./example_dpo_training_output/${SLURM_JOB_ID}
## CONFIG_FILE specifies the training configuration
CONFIG_FILE=../example_configurations/example_dpo_config.yaml

### Run the job locally with torchrun. 
torchrun --standalone --nproc-per-node 8 --no-python fairseq2 lm preference_finetune ${OUTPUT_DIR} --config-file ${CONFIG_FILE} --no-sweep-dir