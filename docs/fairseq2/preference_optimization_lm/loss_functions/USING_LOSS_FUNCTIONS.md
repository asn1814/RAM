# Using Different Loss Functions

## How To
Config files are used to specify loss functions.
- The `criterion` parameter should be set to the name of the loss function given to the `preference_unit_factory`. The pre-implemented loss functions are found at [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/{dpo.py, simpo.py, ...}`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/recipes/lm/preference_finetune) and each contains the decorator `@preference_unit_factory(name)` where `name` is the name of the loss function. Then to specify you want to use SimPO for training, you should set `criterion: simpo` in the config file as `simpo.py` contains an `@preference_unit_factory(simpo)` decorated factory function. This parameter will default to "dpo" if not specified. 
- The `criterion_config` parameter will contain two parts: a `_type_` parameter and a series of other parameters defined by the configuration dataclass associated with the train unit you are using.
    - The `_type_` parameter should be set to the import path of the config dataclass that matches the paramter type of the `criterion` factory function you specified above. The pre-implemented loss functions each contain these configs in their respective files. For example, `simpo.py` contains `SimPOConfig` and so when using `criterion: simpo` you should set `criterion_config: _type_: fairseq2.recipes.lm.preference_finetune.simpo.SimPOConfig`. 
    - The other parameters of `criterion_config` are the parameters of the config dataclass specified in the `_type_` parameter. Continuing the example of SimPO, `SimPOConfig` contains the parameters `beta`, `gamma`, and `nll_scale`, so these can each be parameters of `criterion_config`.

See the examples at [`../example_configurations`](../example_configurations) for more clarity and formatting help. To see how a reference model can be specified for a loss function, refer to [`../example_configurations/example_dpo_config.yaml`](../example_configurations/example_dpo_config.yaml).

## Hyperparameter Setups
Provided are some hyperparameter suggestions for each of the pre-implemented loss functions.

#### DPO
The [DPO Paper](https://arxiv.org/abs/2305.18290) suggests a $\beta$ of 0.1, with a learning rate of 1e-6. On some datasets, $\beta = 0.5$ worked better. SimPO had success iterating over $\beta \in \left[0.01, 0.05, 0.1\right]$. We find that $\beta = 0.1$ with a learning rate of 5.5e-6 and no NLL loss worked well training Llama 3 8B Base from an SFT checkpoint on OpenAssistant2. 

The DPO implementation provides an optional `nll_scale` hyperparameter. This adds NLL loss to the loss function. This has improved performance when used sparingly (around 0.1 or 0.01), especially when beginning DPO training from a base model and not an SFT checkpoint.

`PreferenceOptimizationConfig` also specifies a `mask_source_tokens` parameter. This can affect model training when dropout is used. 

#### SimPO
The [SimPO Paper](https://arxiv.org/abs/2405.14734) suggests a sweep over $\beta \in \left[2.0, 2.5\right]$ and 
$\gamma \in \left[0.3, 0.5, 1.0, 1.2, 1.4, 1.6\right]$. The authors' recommended hyperparameters for different models vary wildly and are presented in Appendix A. For Llama 3 8B Base the authors suggest $\beta = 2.0$, $\gamma = 1.0$, and learning rate 6e-7. For Llama 3 8B Instruct the authors suggest $\beta = 2.5$, $\gamma = 1.4$, and learning rate 1e-6.

We find that SimPO training fails when not started from an SFT checkpoint or with the inclusion of strong NLL loss (we provide an `nll_scale` hyperparameter for this). We had success training Llama 3 8B Base from an SFT checkpoint on OpenAssistant2 with $\beta = 2.0$, $\gamma = 1.0$, learning rate 6e-7, and `nll_scale` 0.

#### ORPO
The [ORPO Paper](https://arxiv.org/abs/2403.07691) suggests $\lambda$ of 0.1 and 0.2 with learning rate 8e-6. The proposed ORPO loss includes NLL loss as a default, so `nll_scale` defaults to 1. 

We had success training Llama 3 8B Base on OpenAssistant2 with $\lambda = 0.2$, learning rate 8e-6, and `nll_scale` 1, but not many hyperparameters were tried so there may be something more effective. 

#### CPO
The [CPO Paper](https://arxiv.org/abs/2401.08417) suggests a $\beta$ of 0.1. The proposed CPO loss includes NLL loss as a default, so `nll_scale` defaults to 1. No learning rate is suggested. 

We had success training Llama 3 8B Base on OpenAssistant2 with $\beta = 0.1$, learning rate 1e-6, and `nll_scale` 0. This outperformed when `nll_scale` was 1, which conflicts with the ablation results of the paper. However, this is a different task and dataset. In addition, not many hyperparameters were tried so there may be something more effective. 