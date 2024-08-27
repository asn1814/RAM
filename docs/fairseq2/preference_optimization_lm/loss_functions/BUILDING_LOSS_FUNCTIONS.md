# Building Loss Functions for LM Preference Finetuning
This guide covers building loss variants for fairseq2's lm preference finetuning recipe. It covers the [SimPO](https://arxiv.org/abs/2405.14734) implementation at `fairseq2/src/fairseq2/recipes/lm/preference_finetune/simpo.py` as a case study. 

### Pre-requisites
To implement your loss function in fairseq2, follow the instructions at `fairseq2/CONTRIBUTING.md` to install the fairseq2 package in editable mode. 

## Overview
fairseq2's registries allow new loss functions to be created in a single file and called from the command line or config file without needing to alter any other code. Loss functions that have already been implemented can be found at `fairseq2/src/fairseq2/recipes/lm/preference_finetune/{dpo.py, simpo.py, ...}`. To create your own, you can just implement another file. 

## How-to

### Components of the loss function
This new file should have the following components.
- A finetune unit class that extends `AbstractTrainUnit[PreferenceOptimizationBatch]`. For the DPO implementation, this is `DPOFinetuneUnit`. This class should override the following functions/properties:
    - `__init__` should initialize the model, parameters, and metric bag. It will be called by the factory function. This will be unique to the hyperparameters used by the loss function.
    - `__call__` should take a `PreferenceOptimizationBatch` and return a `tuple[Tensor, int]` where `Tensor.item()` is the loss on that batch and `int` is the number of targets used to compute the loss and will be used for normalization after the distributed losses are gathered (usually equal to the batch size). **This will be the most important function of the `TrainUnit` and defines the loss function**. 
    >[!TIP]
    >Use the logger to print relevant information each time `__call__` is executed to help sanity check while testing your implementation. Here is an example that can be added to the end of the function: ```log.info(f"Step:{self._step_nr} Rank:{get_rank()} DPO loss: {dpo_loss.item()} ChosenLogp: {chosen_logps[0]} RejectedLogp: {rejected_logps[0]} Seq: {batch.chosen.seqs[0]}")```
    which requires `get_rank` to be imported from `fairseq2.gang`.
    - `set_step_nr` and `metric_bag` should set the current training step number and access the `TrainUnit`'s `SequenceMetricBag`. This will likely be the same across all `TrainUnit`s. 
- A call to `register_metric_formatter` defining how your loss should be named and formatted when logged. 
- A metric bag class that extends `SequenceMetricBag` to log metrics. This should implement a loss value and a function to update the stored loss value that will be called by the train unit class when computing loss. For the DPO implementation, this is `DPOFinetuneMetricBag`.
- A configuration dataclass that defines the parameters used by the loss function. For the DPO implementation, this is `DPOConfig`. This class is used by the configuration file passed to the CLI to specify hyperparameters for training.
- A factory function decorated with the `@preference_unit_factory(name)` decorator where `name` is an appropriate name for the loss. `name` will be what is placed in the `criterion` parameter of the training configuration file to specify training with this loss function. For the DPO implementation, this is `create_dpo_unit`. This should take your config class, model `Module`, root gang `Gang`, and gangs `Mapping[str, Gang]` and return the `TrainUnit[PreferenceOptimizationBatch]` you implemented. If the loss implementation requires a reference model, it should be loaded here (use `fairseq2.recipes.lm.preference_finetune.utils._load_reference_model` for easy loading).

### Using the loss function
You can begin using this new class by simply altering your training configuration file. Set the `criterion` parameter to the `name` given to your `preference_unit_factory`. Set the `criterion_config` parameter's `_type_` parameter to the import path of your configuration dataclass, and its other parameter names to the parameters of that dataclass. The SimPO walkthrough below provides an example. More examples can be found in [`../example_configurations`](../example_configurations).

## SimPO Walkthrough

### Components of the loss function
Consider the loss function proposed as [SimPO](https://arxiv.org/abs/2405.14734) where the loss $\mathcal{L}$ on model $\pi_\theta$ is defined as 
```math
\mathcal{L}_{\text{SimPO}}(\pi_\theta)=-\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}\left[\log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta (y_w | x) - \frac{\beta}{|y_l|} \log \pi_\theta (y_l | x) - \gamma \right) \right]
```
We implement this loss function in `fairseq2/src/fairseq2/recipes/lm/preference_finetune/simpo.py`. It contains the following:
- `SimPOFinetuneUnit` defines the loss. It holds the loss hyperparameters $\beta$ and $\gamma$, as well as an option for NLL loss to be added to the loss computation and the `TrainUnit`'s metric bag. 
    - `__init__` initializes the model, stores the hyperparameters, and initializes the metric bag. 
    - `__call__` computes the loss over a `PreferenceBatch`. For each $y$, `_gather_lprobs` computes $\frac{\log \pi_\theta (y | x)}{|y|}$. `_compute_simpo_loss` takes each and returns the loss $\mathcal{L}_{\text{SimPO}}$. `chosen_output.compute_loss` gets the NLL loss. Then we update the metric bag to record the losses. Deciding how to return the loss and batch size to be summed and normalized across all ranks by the `Trainer` is usually easy but can be tricky. See the section "Normalizing the Loss" below for a breakdown. 
    - `set_step_nr` and `metric_bag` set the current training step number and access the `SimPOFinetuneMetricBag`.
- The loss format is registered with `register_metric_formatter` with name "simpo_loss".
- `SimPOFinetuneMetricBag` stores the loss at each step and provides a function to update the loss. 
- `SimPOConfig` defines the hyparameters `beta`, `gamma`, and `nll_scale` used by `SimPOFinetuneUnit`.
- `create_simpo_unit` is the `preference_unit_factory` with name "simpo"

### Using the loss function
You can use SimPO finetuning by specifying the parameters `criterion` and `criterion_config` of your training configuration file (more info on configs in [`../OVERVIEW.md`](../OVERVIEW.md)) as follows where `beta`, `gamma`, and `nll_scale` can be altered: 
```
criterion: simpo
criterion_config:
  _type_: fairseq2.recipes.lm.preference_finetune.simpo.SimPOConfig
  beta: 2.0
  gamma: 1.0
  nll_scale: 0.0
```
Examples of full configuration files for different loss functions are given in [`../example_configurations`](../example_configurations).

### Normalizing the Loss
If we were only returning SimPO loss and not attempting to add NLL loss to it as well, then normalization would be easy: we would return the sum of SimPO losses from the batch, along with the batch size to normalize by. Let $R$ be the set of ranks, where for any rank $r$, $b_r$ is the total batch size of $r$, $t_r$ is the total tokens of $r$, and $`{\mathcal{L}_{\text{x}}}_r`$ is the total $`\mathcal{L}_{\text{x}}`$ loss of $`r`$. Then the $`\mathcal{L}_{\text{SimPO}}`$ across all ranks would be 
```math
\mathcal{L}_{\text{SimPO}} = \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{SimPO}}}_r}{\displaystyle\sum_{r \in R}b_r}
```
This would be implemented by returning `simpo_loss, chosen_target_batch.batch_size` from `__call__` and works perfectly.

However, we implement functionality to return both the SimPO loss *and* the NLL loss ($`\mathcal{L}_{\text{SimPO}} + \mathcal{L}_{\text{NLL}}`$). Ideally, we want this: 
```math
\mathcal{L}_{\text{SimPO}} + \mathcal{L}_{\text{NLL}} = \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{SimPO}}}_r}{\displaystyle\sum_{r \in R}b_r} + \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{NLL}}}_r}{\displaystyle\sum_{r \in R}t_r}
```

But we can only pass back one loss value from each rank and one number of target value from each rank. This is one solution to approximate that:
```math
\mathcal{L}_{\text{SimPO}} + \mathcal{L}_{\text{NLL}} = \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{SimPO}}}_r + \frac{b_r}{t_r} {\mathcal{L}_{\text{NLL}}}_r}{\displaystyle\sum_{r \in R}b_r} \approx \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{SimPO}}}_r}{\displaystyle\sum_{r \in R}b_r} + \displaystyle\sum_{r \in R}{\frac{1}{|R|t_r} {\mathcal{L}_{\text{NLL}}}_r} \approx \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{SimPO}}}_r}{\displaystyle\sum_{r \in R}b_r} + \frac{\displaystyle\sum_{r \in R}{\mathcal{L}_{\text{NLL}}}_r}{\displaystyle\sum_{r \in R}t_r}
```
This is implemented by returning `simpo_loss + nll_loss * chosen_target_batch.batch_size / chosen_target_batch.num_target_elements(), chosen_target_batch.batch_size` from `__call__`.
This has correct expectation if the batches are random. However, due to dynamic batching, some batches will have a greater number of shorter sequences and some batches will have a lesser number of longer sequences. As a result, this normalization strategy will give more weight to longer sequences due to dynamic batching. 

Another potential strategy is to normalize locally on each rank and then average the ranks. This can be expressed as 
```math
\mathcal{L}_{\text{SimPO}} + \mathcal{L}_{\text{NLL}} = \frac{\displaystyle\sum_{r \in R}\left( \frac{{\mathcal{L}_{\text{SimPO}}}_r}{b_r} + \frac{{\mathcal{L}_{\text{NLL}}}_r}{t_r} \right)}{|R|}
```
This is implemented by returning `simpo_loss / chosen_target_batch.batch_size + nll_loss / chosen_target_batch.num_target_elements(), None` from `__call__`.
This has correct expectation if the batches are random. However, due to dynamic batching, some batches will have a greater number of shorter sequences and some batches will have a lesser number of longer sequences. As a result, this normalization strategy will give more weight to longer sequences due to dynamic batching. 

We choose the first proposed strategy because it guarantees (assuming no numerical error) that the SimPO loss term is normalized optimally. This same normalization strategy is also used by the provided versions of DPO, CPO, and ORPO.
