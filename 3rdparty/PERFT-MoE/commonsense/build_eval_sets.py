#!/usr/bin/env python
"""Export evaluation splits for PERFT-MoE commonsense benchmarks.

This script mirrors the formatting expected by `commonsense/commonsense_evaluate.py`:
for each benchmark it creates `<repo>/commonsense/dataset/<name>/test.json`
containing dictionaries with at least `instruction` and `answer` fields.

NOTE: Updated for datasets>=3.0 which no longer supports trust_remote_code.
Some datasets use parquet conversion path to bypass loading script issues.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, Iterable, Optional, Tuple

from datasets import load_dataset


def load_dataset_compat(dataset_name: str, config: Optional[str], split: str):
    """Load dataset with compatibility for datasets>=3.0.

    Uses parquet conversion path for datasets that have loading scripts.
    """
    # Datasets that need parquet loading due to loading script issues
    PARQUET_DATASETS = {
        "piqa": ("ybisk/piqa", "plain_text"),
        "social_i_qa": ("allenai/social_i_qa", "default"),
        "hellaswag": ("Rowan/hellaswag", "default"),
        "winogrande": ("allenai/winogrande", "winogrande_xl"),
    }

    if dataset_name in PARQUET_DATASETS:
        repo, subset = PARQUET_DATASETS[dataset_name]
        return load_dataset(
            "parquet",
            data_files=f"hf://datasets/{repo}@refs/convert/parquet/{subset}/{split}/*.parquet",
            split="train",
        )

    # Standard loading for other datasets
    if config:
        return load_dataset(dataset_name, config, split=split)
    else:
        return load_dataset(dataset_name, split=split)


def mc_instruction(
    header: str,
    stem: str,
    choices: Iterable[str],
    label_fn: Callable[[int], str],
) -> str:
    """Render a single string containing task instructions plus labelled options."""
    stem = stem.strip()
    option_lines = [
        f"{label_fn(idx)}: {choice.strip()}" for idx, choice in enumerate(list(choices))
    ]
    return "\n".join([header, "", stem, "", "Options:", *option_lines])


def build_boolq(record: Dict) -> Dict:
    labels = ["false", "true"]
    instruction = mc_instruction(
        "Answer the question with `true` or `false`.",
        f"Passage: {record['passage']}\nQuestion: {record['question']}",
        labels,
        lambda idx: labels[idx],
    )
    return {
        "id": record.get("idx", None),
        "instruction": instruction,
        "answer": labels[record["label"]],
    }


def build_piqa(record: Dict) -> Dict:
    label_names = ["solution1", "solution2"]
    instruction = mc_instruction(
        "Choose the more plausible solution (solution1 or solution2).",
        record["goal"],
        [record["sol1"], record["sol2"]],
        lambda idx: label_names[idx],
    )
    return {
        "id": record.get("id", None),
        "instruction": instruction,
        "answer": label_names[int(record["label"])],
    }


def build_social_iqa(record: Dict) -> Dict:
    label_names = {"A": "answer1", "B": "answer2", "C": "answer3", "1": "answer1", "2": "answer2", "3": "answer3"}
    instruction = mc_instruction(
        "Pick the best answer (answer1, answer2, or answer3).",
        f"Context: {record['context']}\nQuestion: {record['question']}",
        [record["answerA"], record["answerB"], record["answerC"]],
        lambda idx: f"answer{idx + 1}",
    )
    return {
        "id": record.get("id", None),
        "instruction": instruction,
        "answer": label_names[record["label"]],
    }


def build_hellaswag(record: Dict) -> Dict:
    label_names = ["ending1", "ending2", "ending3", "ending4"]
    instruction = mc_instruction(
        "Select the option that best completes the situation (ending1-4).",
        f"{record['ctx_a']} {record['ctx_b']}".strip(),
        record["endings"],
        lambda idx: label_names[idx],
    )
    return {
        "id": record.get("ind", None),
        "instruction": instruction,
        "answer": label_names[int(record["label"])],
    }


def build_winogrande(record: Dict) -> Dict:
    label_names = ["option1", "option2"]
    instruction = mc_instruction(
        "Choose the word that correctly fills the blank (option1 or option2).",
        record["sentence"],
        [record["option1"], record["option2"]],
        lambda idx: label_names[idx],
    )
    return {
        "id": record.get("idx", None),
        "instruction": instruction,
        "answer": label_names[int(record["answer"]) - 1],
    }


def build_arc(record: Dict) -> Dict:
    options = list(record["choices"]["text"])
    labels = [f"answer{i + 1}" for i in range(len(options))]
    instruction = mc_instruction(
        "Answer the multiple-choice science question (answer1-5).",
        record["question"],
        options,
        lambda idx: labels[idx],
    )
    label_list = list(record["choices"]["label"])
    answer_idx = label_list.index(record["answerKey"])
    return {
        "id": record.get("id", None),
        "instruction": instruction,
        "answer": labels[answer_idx],
    }


def build_openbookqa(record: Dict) -> Dict:
    options = list(record["choices"]["text"])
    labels = [f"answer{i + 1}" for i in range(len(options))]
    facts_list = record.get("facts")
    facts = " ".join(facts_list) if facts_list else ""
    prompt = (
        f"{facts}\nQuestion: {record['question_stem']}"
        if facts
        else record["question_stem"]
    )
    instruction = mc_instruction(
        "Answer the multiple-choice science question (answer1-4).",
        prompt,
        options,
        lambda idx: labels[idx],
    )
    label_list = list(record["choices"]["label"])
    answer_idx = label_list.index(record["answerKey"])
    return {
        "id": record.get("id", None),
        "instruction": instruction,
        "answer": labels[answer_idx],
    }


DATASET_SPECS: Dict[str, Tuple[str, str, str, Callable[[Dict], Dict]]] = {
    "boolq": ("super_glue", "boolq", "validation", build_boolq),
    "piqa": ("piqa", None, "validation", build_piqa),
    "social_i_qa": ("social_i_qa", None, "validation", build_social_iqa),
    "hellaswag": ("hellaswag", None, "validation", build_hellaswag),
    "winogrande": ("winogrande", "winogrande_xl", "validation", build_winogrande),
    "ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", build_arc),
    "ARC-Easy": ("allenai/ai2_arc", "ARC-Easy", "validation", build_arc),
    "openbookqa": ("allenai/openbookqa", "main", "validation", build_openbookqa),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="dataset",
        help="Base directory under which <dataset>/test.json files are written.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="HF split to export (defaults to validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    for name, (dataset_name, config, default_split, builder) in DATASET_SPECS.items():
        split = args.split if args.split else default_split
        hf_split = load_dataset_compat(dataset_name, config, split)

        formatted = [builder(record) for record in hf_split]
        out_dir = os.path.join(args.output_root, name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "test.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False)

        print(f"Wrote {len(formatted)} records to {out_path}")


if __name__ == "__main__":
    main()
