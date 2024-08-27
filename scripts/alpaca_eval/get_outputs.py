# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import datasets
import fire
from vllm import LLM, SamplingParams

LLAMA3_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
EMPTY_TEMPLATE = "{instruction}"
TEMPLATE_DICT = {
    "llama3_template": LLAMA3_TEMPLATE,
    "empty_template": EMPTY_TEMPLATE,
}


def get_generations(
    model_file: str = None,
    model_name: str = None,
    outputs_file: str = None,
    default_reference: bool = False,
    version: Literal[
        "alpaca_eval", "alpaca_eval_gpt4_baseline"
    ] = "alpaca_eval_gpt4_baseline",
    template: str = "empty_template",
    dtype: Literal["half", "auto"] = "half",
):
    """Generates completions for AlpacaEval.

    Args:
        model_file (str, optional): The path to the HF model to get generations from. Defaults to None.
        model_name (str, optional): The name of the model to label the output with. Defaults to None.
        outputs_file (str, optional): Completions are written to outputs_file. The path to the file location must exist. Defaults to None.
        default_reference (bool, optional): If true, returns baseline outputs to be used as a comparison. Defaults to False.
        version (Literal[&quot;alpaca_eval&quot;, &quot;alpaca_eval_gpt4_baseline&quot;], optional): The baseline dataset to return if default_reference is true. Defaults to "alpaca_eval_gpt4_baseline", which is used for AlpacaEval2.
        template (str, optional): The template to wrap the prompt with. Defaults to an empty template.
        dtype (Literal[&quot;half&quot;, &quot;auto&quot;], optional): The datatype to use for vllm inference. Defaults to "half".
    """
    template = TEMPLATE_DICT[template]
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", version)["eval"]
    if default_reference:
        eval_set.to_pandas().to_json(outputs_file, orient="records", mode="w")
        return

    model = model_file  # HF model
    tensor_parallel = 4  # up to max GPUs in the machine

    llm = LLM(
        model,
        tensor_parallel_size=tensor_parallel,
        dtype=dtype,
        distributed_executor_backend="ray",
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.9,
        max_tokens=4096,
        seed=1814,
        # top_k=args.top_k,
        # repetition_penalty=args.repetition_penalty,
        top_p=1,
        use_beam_search=False,
        stop_token_ids=None,
    )

    prompts = []
    for example in eval_set:
        prompts.append(template.format(instruction=example["instruction"]))

    outputs = llm.generate(prompts, sampling_params)

    completions = [output.outputs[0].text for output in outputs]
    model_names = [model_name for _ in outputs]
    finish_reasons = [output.outputs[0].finish_reason for output in outputs]

    # print how many completions were terminated due to max_tokens
    count = 0
    for output in outputs:
        if output.outputs[0].finish_reason == "length":
            count += 1
    print(
        f"{count} of {len(outputs)} generations ({(count/len(outputs))*100:.2f}%) finished due to max length"
    )

    eval_set = eval_set.remove_columns(["output"])
    eval_set = eval_set.add_column("output", completions)
    eval_set = eval_set.remove_columns(["generator"])
    eval_set = eval_set.add_column("generator", model_names)
    eval_set = eval_set.add_column("finish_reason", finish_reasons)

    eval_set.to_pandas().to_json(outputs_file, orient="records")


# Fire integration
if __name__ == "__main__":
    fire.Fire(get_generations)

"""
python get_outputs.py --outputs_file outputs/default_generations.json --default_reference True
python ${GET_OUTPUTS_PY} --model_file ${HF_LLAMA3_MODEL_PATH} --model_name ${MODEL_NAME} --outputs_file ${OUTPUTS_PATH} --dtype half
"""
