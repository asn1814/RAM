## Conda Env Installation
- You will need one virtual environment with vllm. 
- You will need one virtual environment with alpaca_eval installed. Use the provided [`setup_scripts/h2_alpacaeval_conda_install.sh`](setup_scripts/h2_alpacaeval_conda_install.sh) script to do this automatically. 
- For inference with V100s, see the section below. 

# Evaluation with AlpacaEval2
We cover a way to get quick and cheap evaluations with AlpacaEval2 locally instead of with OpenAI credits. This results in a worse benchmark but is enough to show that a model is improving. For a full evaluation with the best annotator, refer to the [documentation](https://github.com/tatsu-lab/alpaca_eval/tree/main?tab=readme-ov-file).

This assumes you begin with models in HuggingFace format that you want to evaluate against each other.

> [!IMPORTANT]
> To try out these scripts, just run `sbatch examples/evaluate_example.sh` with the `alpacaeval` conda environment above activated.

## Collect output data
For each evaluation, you need reference outputs and model outputs. Both can be generated with `python get_outputs.py --model_file <hf_model_path> --model_name <model_name> --outputs_file <model_outputs_path> --template <template>` in an environment with vllm. To generate the default reference outputs, use `python get_outputs.py --default_reference True --outputs_file outputs/default_outputs.json`. It is reccomended to use a template to wrap the raw prompts so that the special tokens match the training data. An example template for the Llama 3 family of models is provided in `get_outputs.py` and can be used with `--template llama3_template`.

> [!NOTE]
> By default, the reference outputs are the `gpt4_turbo` outputs for AlpacaEval2 set. This can be toggled with the `--version` flag of `get_outputs.py`.

## Annotating with Llama 3.1 70b Instruct
Config files to annotate and evaluate models using Llama 3.1 70b Instruct, Llama 3.1 8b Instruct, and Llama 3 8b Instruct are provided. To use a different model as an annotator/evaluator, see the section on creating a new evaluator below. **For Llama 3.1 70b Instruct, run on 8 V100 GPUs with 32GB RAM each (`--constaint=volta32gb`)to prevent OOM.**

> [!NOTE]
> The provided Llama 3.1 70b Instruct annotator running locally on V100s gets 68.56 human agreement (humans get 65.66 human agreement), which makes it better than human annotators and nearly reaches the best GPT4 performance. Llama 3.1 8b Instruct gets 65.09 human agreement, and Llama 3 8b Instruct gets 63.08 human agreement. However, there is some indication that these annotators may favor Llama-based models. 

For any annotator, we create a leaderboard that tracks how well various models perform against each other when evaluated by that annotator. To evaluate outputs with the annotator, use 
```bash
alpaca_eval make_leaderboard \
  --leaderboard_path <path_to_save_leaderboard> \
  --all_model_outputs <model_outputs_path> \
  --reference_outputs <reference_outputs_path> \
  --annotators_config <path_to_annotator_config.yaml>
```

where:
- `leaderboard_path`: path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will append.
- `all_model_outputs`: The json path to the outputs of all models to add to the leaderboard (as a single file or by globbing multiple files). Each dictionary should contain the keys (`instruction` and `output`) that are formatted in the prompts and a column `generator` with the name of the current model.
- `reference_outputs`: the path to the outputs of the reference model. Each dictionary should contain the keys (`instruction` and `output`) that are formatted in the prompts. By default, the reference outputs are the 003 outputs on AlpacaEval set.
- `annotators_config`: The path to the annotator's config file. Llama 3 8b Instruct, Llama 3.1 8b Instruct, and Llama 3.1 70b Instruct are provided in `evaluator_configs/`. 

This command can be called repeatedly with different `model_outputs_path` and it will append to the leaderboard. The other three parameters should remain consistent across model evaluations. 

## Creating a new evaluator
Full documentation to create your own annotator can be found [here](https://github.com/tatsu-lab/alpaca_eval/tree/main?tab=readme-ov-file). For something quick, I recommend turning the sampling temperature to 0 and then creating the prompt such that the exact output needed by the `fn_completion_parser` specified in your config will be produced. It is helpful to set `max_tokens` so that generation will terminate immediately after the correct output is produced.

## Inference with V100s 
Inference is very fast with V100s, so it is useful to infer on them rather than more expensive GPUs. However, VLLM is not automatically set up to handle float16, so inputs need to change. The fix is easy:
- The `--dtype` flag of `get_outputs.py` should be set to `half`.  
- In line `dtype: "auto"` of your annotator's `configs.yaml`, `dtype="auto"` should be replaced with `dtype="half`. You may also need to tweak `tensor_parallel_size` to fully utilize the correct number of GPUs. The Llama 3.1 `config.yaml` is already set for this, and should be switched to `auto` for inference on better GPUs. 
Sit back and enjoy inference on V100s!