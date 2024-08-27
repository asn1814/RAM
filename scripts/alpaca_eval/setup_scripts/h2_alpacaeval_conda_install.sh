#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

env_name=alpacaeval

env_path=/$env_name # CHANGE ME TO WHERE YOU WANT TO INSTALL ENVIRONMENT

if [[ -d "$env_path" ]]; then
    echo "Directory \"$env_path\" already exists!" >&2

    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Conda and make sure that it is available in your PATH." >&2

    exit 1
fi

mkdir --parents "$env_path"

cuda_version=12.1
torch_version=2.3.0

echo "Creating '$env_name' Conda environment with PyTorch $torch_version and CUDA $cuda_version..."

conda create\
    --prefix "$env_path/alpacaeval"\
    --yes\
    --strict-channel-priority\
    --override-channels\
    --channel pytorch\
    --channel nvidia\
    --channel conda-forge\
    python==3.10.0\
    pytorch==$torch_version\
    pytorch-cuda==$cuda_version\
    libsndfile==1.0.31

# sym link prefix based environment to access it using the alias name
envs_dirs=$(conda info --json | jq ".envs_dirs[0]" | xargs echo)
ln -s "$env_path/alpacaeval" "$envs_dirs/$env_name"
echo "Symlink $env_path/alpacaeval -> $envs_dirs/$env_name was created"

echo "Installing alpaca_eval"

conda run --prefix "$env_path/alpacaeval" --no-capture-output --live-stream\
    pip install alpaca_eval[all] --pre --upgrade\

cd "$env_path"

cat << EOF

Done!
To activate the environment, run 'conda activate $env_path/alpacaeval'.
EOF
