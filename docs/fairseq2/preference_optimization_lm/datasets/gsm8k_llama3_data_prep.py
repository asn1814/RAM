# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from pathlib import Path

import yaml
from fire import Fire
from tqdm import tqdm
from utils import built, mark_done, save_to_jsonl
from vllm import LLM, SamplingParams

from datasets import load_dataset

L3_START_INST = "<|start_header_id|>user<|end_header_id|>\n\n"
L3_END_INST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
FEWSHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8",
    },
]

# utils from gsm8k paper to extract a correct answer and match it to the prompt
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def prep_data(data_dir: str, card_dir: str, dtype: str = "auto"):
    """Prepares a dataset of positive/negative completions from GSM8k using Llama 3 8b Base to generate negative completions.

    Args:
        data_dir (str): The directory the raw data `.jsonl` will be written to.
        card_dir (str): The directory the `gsm8k_preference_llama3.yaml` asset card will be written to.
        dtype (str, optional): Data type for model weights and activations. Defaults to "auto".
    """
    # setup fewshot
    fewshot = ""
    for ex in FEWSHOT:
        fewshot += f"{ex['question']} {ex['answer']} "

    # load and build prompts
    data = load_dataset("gsm8k", "main")
    gsm8k_train = []
    for ex in data["train"]:
        example = {"input": ex["question"], "label": ex["answer"]}
        gsm8k_train.append(example)

    _prompts = []
    for example in gsm8k_train:
        _prompts.append(f"{fewshot}{example['input']} ")

    model = "dummy/path"  # filepath to Llama 3.1 8B in HuggingFace format here

    # setup model
    tensor_parallel = 8  # up to max GPUs in the machine
    llm = LLM(
        model, trust_remote_code=True, tensor_parallel_size=tensor_parallel, dtype=dtype
    )
    sampling_params = SamplingParams(
        n=5,
        temperature=1,
        top_p=1,
        max_tokens=256,
        seed=1814,
        # top_k=args.top_k,
        # repetition_penalty=args.repetition_penalty,
        use_beam_search=False,
    )

    outputs = llm.generate(_prompts, sampling_params)

    # get answers that are well-formatted but ultimately incorrect
    _processed_samples = []
    total = 0
    suitable = 0
    for output, example in tqdm(zip(outputs, gsm8k_train), "finding negative examples"):
        for generation in output.outputs:
            total += 1
            processed_generation = re.split(
                re.compile(r"(#### (\-?[0-9\.\,]+))"), generation.text, maxsplit=1
            )  # get up to the first answer
            if (
                len(processed_generation) == 1
            ):  # didn't find the regex, so answer not formatted correctly
                continue
            processed_generation = processed_generation[0] + processed_generation[1]
            if (
                extract_answer(processed_generation) == INVALID_ANS
            ):  # unparseable answer
                continue
            if not is_correct(
                processed_generation, example[0]["label"]
            ):  # answer is incorrect -> add it
                train_ex = {
                    "src": f"{L3_START_INST} {example[0]['input']} {L3_END_INST}",
                    "tgt_chosen": f"{example[0]['label']}",
                    "tgt_rejected": f"{processed_generation}",
                }
                _processed_samples.append(train_ex)
                suitable += 1
                break

    print(f"Out of {total} generations, {suitable} were suitable.")

    # write data

    version_string = "gsm8k_llama_3_8b_negatives"
    save_dir = Path(data_dir)
    save_dir.resolve().mkdir(parents=True, exist_ok=True)
    if built(save_dir, version_string):
        print(
            f"Data version {version_string} exists in {save_dir}, skipping data writing"
        )
    else:
        save_to_jsonl(_processed_samples, f"{data_dir}/data.chunk.0000.jsonl")
        mark_done(save_dir, version_string)

    # create dataset card
    name = "gsm8k_preference_llama3"
    card = [
        {"name": f"{name}", "dataset_family": "generic_preference_optimization"},
        {"name": f"{name}@faircluster", "data": f"{Path(data_dir).resolve()}"},
        # implement @awscluster here
    ]

    # write card to .yaml file
    Path(card_dir).mkdir(parents=True, exist_ok=True)
    with open(
        f"{card_dir}/gsm8k_preference_llama3.yaml", mode="+w", encoding="utf-8"
    ) as file:
        yaml.dump_all(card, file, sort_keys=False)


def load(split="train"):
    data = load_dataset("gsm8k", "main")
    dataset = []
    for ex in data[split]:
        example = {"input": ex["question"], "label": ex["answer"]}
        dataset.append(example)
    return dataset


# entrypoint
if __name__ == "__main__":
    Fire(prep_data)

"""
python gsm8k_llama3_data_prep.py --data_dir ./gsm8k_data --card_dir ./cards --dtype half
"""
