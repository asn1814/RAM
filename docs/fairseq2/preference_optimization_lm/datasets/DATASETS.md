# Datasets for Preference Optimization
Preference finetuning requires a dataset that consists of $(x, y_w, y_l)$ pairs where $x$ is a prompt with preferred completion $y_w$ and dispreferred completion $y_l$. This guide covers how to build and use preference optimization datasets with fairseq2. 

## How Datasets Work
Data is loaded into the training recipe in the following manner:
1. fairseq2 reads data from `.jsonl` files that can be passed either as an asset card or a filepath in the training configuration's `dataset` parameter. This occurs in [`fairseq2/src/fairseq2/recipes/lm/preference_finetune/recipe.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/recipes/lm/preference_finetune/recipe.py).
2. This data is loaded by [`fairseq2/src/fairseq2/datasets/preference.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/datasets/preference.py) which contains:
    - The abstract class `PreferenceOptimizationDataset` that defines the parameters for batching preference optimization datasets.
    - The class `GenericPreferenceOptimizationDataset` which expects a JSON file of objects each with the keys `src`, `tgt_chosen`, and `tgt_rejected` (corresponding to $x$, $y_w$, and $y_l$ respectively) and the optional key `id`. This class implements how to create a `DataPipelineReader` for a `PreferenceOptimizationDataset`. This includes tokenization and batching.
3. The `DataPipelineReader` is passed to the `Trainer` and training begins.

>![TIP]
>Specify that a dataset is a `GenericPreferenceOptimizationDataset` by setting the asset file's `dataset_family` parameter to `generic_preference_optimization`.

## Training Dataset Setup
`GenericPreferenceOptimizationDataset` provides the basic functionality needed for most preference optimization tasks: `src`, `tgt_chosen`, and `tgt_rejected` keys. If you require more than this, see the section "Creating New Types of Datasets" below. 

Tips for building `GenericPreferenceOptimizationDataset` datasets:
- How the dataset will be tokenized by [`fairseq2/src/fairseq2/datasets/preference.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/datasets/preference.py) depends on the tokenizer used. You can find the model family tokenizers in [`fairseq2/src/fairseq2/models`](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models).
- For Llama 3 models, fairseq2 concatenates `src` + `tgt_chosen` and `src` + `tgt_rejected`. It adds the "end of text" special token to the end of each sequence and the "begin of text" special token to the beginning of each sequence. This means:
    - Don't add "begin of text" or "end of text" to the start/end of `tgt_chosen` or `tgt_rejected`.
    - Do add any other special tokens in the appropriate places. For example, add Llama 3 Instruct role and message special tokens. 

Examples of dataset preparation scripts are provided:
- `openassistant2_data_prep.py` loads OpenAssistant2 and uses the provided response rankings to create chosen/rejected pairs.
- `gsm8k_data_prep.py` uses Llama 3 8B and vLLM to generate incorrect answers to GSM8k which are paired with the provided gold labels to create a preference dataset. 

## Creating New Types of Datasets
Some preference optimization tasks need additional functionality that `GenericPreferenceOptimizationDataset` doesn't provide. For example, beta-DPO requires a "beta" key, and KTO requires a "good/bad" key. To implement a dataset with this functionality, follow these steps:
- Extend `GenericPreferenceOptimizationDataset`. Overwrite `create_reader`, changing `cat_source_and_target` to return the necessary field. An example of this is the `id` field.
- Add a corresponding `DatasetLoader` and register it with `load_preference_optimization_dataset` (creating a new family name). See how this is done for `GenericPreferenceOptimizationDataset` in [`fairseq2/src/fairseq2/datasets/preference.py`](https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/datasets/preference.py).
- Build your dataset with the new field, and specify the correct new dataset family in the asset card.
- Run training!
