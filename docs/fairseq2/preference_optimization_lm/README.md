# Preference Optimization Finetuning for Language Models with fairseq2
Preference optimization is a finetuning strategy that uses a dataset of $(x, y_w, y_l)$ pairs where $x$ is a prompt, $y_w$ is a preferred completion for that prompt, and $y_l$ is a dispreferred completion for that prompt. fairseq2 provides recipes to finetune language models using these datasets and various preference optimization criteria. Finally, fairseq2 provides a way to extend the recipe with your own optimization criterion with minimum efforts. 

### Pre-requisites and Installation

**Environment:** You should have fairseq2 installed and your fairseq2 environment ready. Follow the installation guide in fairseq2 repo. 

**Hardware:** Experiments described below were tested using 8-gpu node with Nvidia A100 GPUs.

## Contents
- [The Overview](OVERVIEW.md): a high level guide that walks through how everything fits together. Read this to quickly gain a better general understanding of the preference optimization recipes.
- [The Building Loss Functions Guide](loss_functions/BUILDING_LOSS_FUNCTIONS.md): instructions and advice to implement and use your own loss functions. 
- [The Getting Results Guide](loss_functions/USING_LOSS_FUNCTIONS.md): how to specify training with different loss functions (**including hyperparameter suggestions**) and replicate our results.
- [The Datasets Guide](datasets/DATASETS.md): how to build fairseq2 datasets for preference optimization.
- [Example Configs](example_configurations/): example configuration files to specify training parameters.
- [Example Training Scripts](example_training_scripts/): example training scripts to begin training locally or with SLURM. 
- [Example Dataset Scripts](datasets/): example dataset preparation scripts to create data files and asset cards.

## Get Started
Once you've set up and activated your fairseq2 environment, you can get started immediately by running [quick-start.sh](./quick-start.sh). This creates a preference dataset from OpenAssistant2 and begins trainining Llama 3 8B with DPO. 
