#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### SBATCH directives to specify job configuration
## %j is the job id, %u is the user id. These files are where output is printed.
#SBATCH --output=/checkpoint/%u/jobs/example_dpo_fs2-%j.out
#SBATCH --error=/checkpoint/%u/jobs/example_dpo_fs2-%j.err
#SBATCH --job-name=DPO
## 1 node, 8 ampere80gb GPUs each with 10 CPUs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=8
#SBATCH --constraint=ampere80gb
#SBATCH --partition=dummy_partition
#SBATCH --time 05:00:00

### Setting environment variables for the job.
## OUTPUT_DIR specifies the location that the trained model and training metrics should be stored.
OUTPUT_DIR=./example_dpo_training_output/${SLURM_JOB_ID}
## CONFIG_FILE specifies the training configuration
CONFIG_FILE=../example_configurations/example_dpo_config.yaml

### Run the job. 
srun fairseq2 lm preference_finetune ${OUTPUT_DIR} --config-file ${CONFIG_FILE} --no-sweep-dir