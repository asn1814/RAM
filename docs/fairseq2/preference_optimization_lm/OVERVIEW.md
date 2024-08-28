# Overview of Preference Optimization Finetuning for Language Models with fairseq2

## How to train
The entrypoint for preference finetuning for language models with fairseq2 is the command `fairseq2 lm preference_finetune <output_dir>` where `<output_dir>` is the output directory where the training run's model checkpoints, metric logs, and GPU logs will be stored. Useful flags:
- `--config-file <config_file>`, where `<config_file>` is a `.yaml` file containing the specification for a training configuration. This is the easiest and most thorough way to train. 
- `--preset <preset>`, where `<preset>` is one of the preset strings defined in [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/recipe.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/recipes/lm/preference_finetune/recipe.py), which allows the use of a pre-built preset for training. 

## The important files
Preference finetuning relies on 3 main files that define the training process. 
- [`fairseq2/src/fairseq2/datasets/preference.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/datasets/preference.py) loads the preference finetuning dataset. 
- [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/{dpo.py, simpo.py, ...}`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/recipes/lm/preference_finetune) each define a different type of TrainUnit that is used to calculate the loss and log training metrics. Only one is used at a time. 
- [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/recipe.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/recipes/lm/preference_finetune/recipe.py) loads the TrainUnit, optimizer, learning rate scheduler, and dataset to create the overall Trainer that handles training.

## Data
Preference finetuning requires a dataset that consists of $(x, y_w, y_l)$ pairs where $x$ is a prompt with preferred completion $y_w$ and dispreferred completion $y_l$. fairseq2 reads data from `.jsonl` files that can be passed either as an asset card or a filepath in the configuration's `dataset` parameter. 

This file is loaded by [`fairseq2/src/fairseq2/datasets/preference.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/datasets/preference.py), which expects a JSON file of objects each with the keys `src`, `tgt_chosen`, and `tgt_rejected` (corresponding to $x$, $y_w$, and $y_l$ respectively) and the optional key `id`. 

>[!IMPORTANT]
>fairseq2 **does not** add special formatting tokens around the `src` input, so if your model expects special tokens these should be added when you create the JSON file. fairseq2 **does** add an end-of-text token to the end of the input sequence, so this special token should not be added to the end of `tgt_chosen` or `tgt_rejected` in the JSON file. 

See [`DATASETS.md`](datasets/DATASETS.md) for more details on how to construct datasets.

## Configs
Configuration files for training can be provided to fairseq2 through the `--config_file` flag. The configuration parameters are defined by the `PreferenceOptimizationConfig` class in [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/recipe.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/recipes/lm/preference_finetune/recipe.py). 

Parameters "`criterion`" and"`criterion_config`" are unique to preference_optimization and define the loss function that will be used. 
- The `criterion` parameter must be a name registered with the `preference_unit_factory`. Examples of this are in the files [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/{dpo.py, simpo.py, ...}`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/recipes/lm/preference_finetune). 
- The `_type_` parameter of `criterion_config` must be the path to the corresponding config DataClass (`DPOConfig, SimPOConfig, ...`). The remaining parameters of `criterion_config` can then be the parameters of that config. For a clear example of this, see `example_dpo_config.yaml`. 

## Loss Functions
The loss function used by the preference finetune Trainer is defined by the `criterion`/`criterion_config` parameters, as they ultimately select the TrainUnit that defines a loss function for training. fairseq2 provides implementations of published loss functions such as DPO, SimPO, CPO, and ORPO. Each of [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/{dpo.py, simpo.py, ...}`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/recipes/lm/preference_finetune) defines a loss function in its `TrainUnit`'s `__call__` function and corresponding hyperparameters in its config class.  

See [`example_configurations`](example_configurations/) for examples of how to specify the loss function to use.

## Training From A Checkpoint
You may want to begin preference finetuning *after* supervised finetuning has already occurred. fairseq2 provides an easy way to load models from checkpoints with the `--config_file` flag. The `resume_checkpoint_dir` parameter allows a fairseq2 model to be loaded when training is initialized with the checkpoint specified by the `model` field. 

>[!NOTE]
>If the loss function you are using requires a reference model, don't forget that you must also specify if you want it to be loaded from a checkpoint as well. 

See [`example_dpo_from_checkpoint_config.yaml`](example_configurations/example_dpo_from_checkpoint_config.yaml)

## Creating New Loss Functions
fairseq2 has been built to be easily extensible. For an in-depth guide covering how to make and experiment with new loss functions, see the provided [`loss_functions/BUILDING_LOSS_FUNCTIONS.md`](loss_functions/BUILDING_LOSS_FUNCTIONS.md) tutorial.