# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from pathlib import Path
from string import Template

import yaml
from fire import Fire
from tqdm import tqdm

from datasets import load_dataset
from utils import built, map_str_to_uuid, mark_done, save_to_jsonl

L3_START_INST = "<|start_header_id|>user<|end_header_id|>\n\n"
L3_END_INST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def prep_data(data_dir: str, card_dir: str):
    """Prepares a dataset of positive/negative completions from OpenAssistant2 and a corresponding instruction tuning dataset.

    Args:
        data_dir (str): The directory the raw data `.jsonl` will be written to.
        card_dir (str): The directory the `oasst2_preference_llama3.yaml` asset card will be written to.
    """
    # create dataset card
    name = "openassistant2_llama3"
    card = []

    # load dataset, get preference pairs
    oasst2_splits = load_openeft_splits()

    for split_name, data in oasst2_splits.items():
        _processed_samples = []
        for ex_g in tqdm(data, f"Processing split {split_name}"):
            ranks = [ex["rank"] for ex in ex_g]
            if 0 not in ranks:
                continue
            chosen_i = ranks.index(0)
            for rejected_i in range(
                len(ranks)
            ):  # combine 0th rank with everything else
                if rejected_i == chosen_i:
                    # skip when negative == positive
                    continue
                train_ex = {
                    "src": f"{L3_START_INST} {ex_g[chosen_i]['input']} {L3_END_INST}",
                    "tgt_chosen": f"{ex_g[chosen_i]['label']}",
                    "tgt_rejected": f"{ex_g[rejected_i]['label']}",
                }
                _processed_samples.append(train_ex)

        # write data
        version_string = f"{name}_preference_{split_name}"
        save_dir = Path(f"{data_dir}/preference/{split_name}")
        save_dir.resolve().mkdir(parents=True, exist_ok=True)
        if built(save_dir, version_string):
            print(
                f"Data version {version_string} exists in {save_dir}, skipping data writing"
            )
        else:
            save_to_jsonl(
                _processed_samples,
                f"{data_dir}/preference/{split_name}/data.chunk.0000.jsonl",
            )
            mark_done(save_dir, version_string)

        # add path to card
        card.append(
            {
                "name": f"{name}_preference_{split_name}",
                "dataset_family": "generic_preference_optimization",
            }
        )
        card.append(
            {
                "name": f"{name}_preference_{split_name}@faircluster",
                "data": str(Path(f"{data_dir}/preference/{split_name}").resolve()),
            }
        )
        # implement path in @awscluster here

        # make instruction dataset
        _instruction_samples = []
        seen = set()
        prevent_duplicates = False
        for ex in _processed_samples:
            new_ex = {"src": ex["src"], "tgt": ex["tgt_chosen"]}
            new_ex["id"] = map_str_to_uuid(new_ex["src"] + new_ex["tgt"])
            if prevent_duplicates:
                if new_ex["id"] not in seen:
                    _instruction_samples.append(new_ex)
                    seen.add(new_ex["id"])
            else:
                _instruction_samples.append(new_ex)

        # write data
        version_string = f"{name}_instruction_{split_name}"
        save_dir = Path(f"{data_dir}/instruction/{split_name}")
        save_dir.resolve().mkdir(parents=True, exist_ok=True)
        if built(save_dir, version_string):
            print(
                f"Data version {version_string} exists in {save_dir}, skipping data writing"
            )
        else:
            save_to_jsonl(
                _instruction_samples,
                f"{data_dir}/instruction/{split_name}/data.chunk.0000.jsonl",
            )
            mark_done(save_dir, version_string)

        # add path to card
        card.append(
            {
                "name": f"{name}_instruction_{split_name}",
                "dataset_family": "generic_instruction",
            }
        )
        card.append(
            {
                "name": f"{name}_instruction_{split_name}@faircluster",
                "data": str(Path(f"{data_dir}/instruction/{split_name}").resolve()),
            }
        )
        # implement path in @awscluster here

    # write card to .yaml file
    Path(card_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{card_dir}/{name}.yaml", mode="+w", encoding="utf-8") as file:
        yaml.dump_all(card, file, sort_keys=False)


def load_openeft_splits(rank: int = None, first_turn_only: bool = False):
    """Prepares the train/dev partitions w reproducible shuffling

    Args:
        rank (int): load samples with these rank in responses, see code in process_dataset
        first_turn_only (bool): load only samples that correspons to first turn in the conversation

    Returns:
        Dict: {"train": Dataset, "dev": Dataset} dict with dataset partitions
    """
    original_train_split = process_dataset(split="train", language="en", rank=rank)
    if first_turn_only:
        filtered_data = []
        for ex_g in original_train_split:
            if ex_g[0]["parent_id"] == ex_g[0]["message_tree_id"]:
                filtered_data.append(ex_g)
    else:
        filtered_data = original_train_split

    random.Random(333).shuffle(filtered_data)
    # leaving last 500 items of deterministically shuffled dataset
    return {"train": filtered_data[:-500], "dev": filtered_data[-500:]}


def process_dataset(
    split: str = "train",
    multiturn: bool = False,
    language: str | None = None,
    rank: str | None = None,
    user_assistant_template: str | None = None,
):
    data = load_dataset("OpenAssistant/oasst2")
    dataset = []
    template = Template(user_assistant_template)

    if split not in data.keys():
        raise ValueError(
            f"{split} split is not supported. Available splits are: {data.keys()}"
        )

    example_groups = {}
    parents = {}
    for ex in data[split]:
        parents[ex["message_id"]] = {
            "text": ex["text"],
            "parent_id": ex["parent_id"],
        }

        if language is not None and ex["lang"] != language:
            # some examples have wrong attributes; in particular, plenty (>8) trees have
            # issues with 1 node having incorrect attribute (language).
            # we will keep them as parents to preserve trees, but won't add to groups
            # These are outliers and should have only 1 item in a group. skip these
            continue

        # create groups by common context
        if ex["role"] == "assistant":
            example = {
                "label": ex["text"],
                "role": ex["role"],
                "message_id": ex["message_id"],
                "parent_id": ex["parent_id"],
                "message_tree_id": ex["message_tree_id"],
                "lang": ex["lang"],
                "review_count": ex["review_count"],
                "review_result": ex["review_result"],
                "deleted": ex["deleted"],
                "rank": ex["rank"],
                "detoxify": ex["detoxify"],
                "emojis": ex["emojis"],
                "labels": ex["labels"],
            }
            if example["parent_id"] not in example_groups:
                example_groups[example["parent_id"]] = [example]
            else:
                example_groups[example["parent_id"]].append(example)
        elif ex["role"] != "prompter":
            raise ValueError(f"Role `{ex['role']}` is not supported")

    for group_id, example_group in example_groups.items():
        # add context
        context = parents[group_id]["text"]
        if multiturn:
            context = template.substitute(
                {"user_prompt": context, "assistant_prompt": ""}
            )

            cur_id = parents[group_id]["parent_id"]
            while cur_id is not None:
                # add all ancestors untill reach root
                context = (
                    template.substitute(
                        {
                            "user_prompt": parents[parents[cur_id]["parent_id"]][
                                "text"
                            ],
                            "assistant_prompt": parents[cur_id]["text"],
                        }
                    )
                    + "\n"
                    + context
                )
                cur_id = parents[parents[cur_id]["parent_id"]]["parent_id"]
        for ex in example_group:
            ex["input"] = context
        if rank is not None:
            # similar to the top-k ranking in
            # https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/custom_datasets/oasst_dataset.py#L74
            filtered_group = []
            # keep None ranks only for parents and single children
            candidates = {}
            found_children = set()
            for ex in example_group:
                if ex["role"] == "assistant":
                    if ex["rank"] is None:
                        candidates[ex["parent_id"]] = ex
                    elif ex["rank"] == rank:
                        filtered_group.append(ex)
                        found_children.add(ex["parent_id"])
                else:
                    filtered_group.append(ex)
                for par_id, cand in candidates.items():
                    if par_id not in found_children:
                        filtered_group.append(cand)
                        found_children.add(par_id)
            if len(filtered_group) > 0:
                dataset.append(filtered_group)
        else:
            dataset.append(example_group)

    return dataset


# entrypoint
if __name__ == "__main__":
    Fire(prep_data)

"""
python openassistant2_llama3_data_prep.py --data_dir ./oasst2_data --card_dir ./cards
"""
