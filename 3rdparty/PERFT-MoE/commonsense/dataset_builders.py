"""Dataset builders for PERFT-MoE commonsense benchmarks.

Each function returns a list of records formatted for `finetune.py`:
    {
        "instruction": str,
        "input": str,
        "output": str,
    }
The registry below exposes the train split for each benchmark so scripts can
export them individually or as a merged corpus.

NOTE: Updated for datasets>=3.0 which no longer supports trust_remote_code.
Some datasets (piqa, social_i_qa) now use alternative HuggingFace repos.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from datasets import load_dataset


def _format_options(prefixes: Sequence[str], choices: Sequence[str]) -> str:
    return "\n".join(
        f"{prefixes[idx]}: {choices[idx].strip()}" for idx in range(len(choices))
    )


def _build_input(stem: str, prefixes: Sequence[str], choices: Sequence[str]) -> str:
    stem = stem.strip()
    return "\n".join([stem, "", "Options:", _format_options(prefixes, choices)])


def build_boolq(split: str = "train") -> List[dict]:
    dataset = load_dataset("super_glue", "boolq", split=split)
    prefixes = ["false", "true"]
    header = "Answer the question with `true` or `false`."
    records: List[dict] = []
    for example in dataset:
        stem = f"Passage: {example['passage']}\nQuestion: {example['question']}"
        records.append(
            {
                "instruction": header,
                "input": _build_input(stem, prefixes, prefixes),
                "output": prefixes[int(example["label"])],
            }
        )
    return records


def build_piqa(split: str = "train") -> List[dict]:
    # Load from parquet conversion to avoid loading script issue in datasets>=3.0
    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/ybisk/piqa@refs/convert/parquet/plain_text/{split}/*.parquet",
        split="train",  # parquet loader uses "train" as default split name
    )
    prefixes = ["solution1", "solution2"]
    header = "Choose the more plausible solution (solution1 or solution2)."
    records: List[dict] = []
    for example in dataset:
        choices = [example["sol1"], example["sol2"]]
        records.append(
            {
                "instruction": header,
                "input": _build_input(example["goal"], prefixes, choices),
                "output": prefixes[int(example["label"])],
            }
        )
    return records


def build_social_iqa(split: str = "train") -> List[dict]:
    # Load from parquet conversion to avoid loading script issue in datasets>=3.0
    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/allenai/social_i_qa@refs/convert/parquet/default/{split}/*.parquet",
        split="train",  # parquet loader uses "train" as default split name
    )
    prefixes = ["answer1", "answer2", "answer3"]
    header = "Pick the best answer (answer1, answer2, or answer3)."
    records: List[dict] = []
    for example in dataset:
        choices = [example["answerA"], example["answerB"], example["answerC"]]
        stem = f"Context: {example['context']}\nQuestion: {example['question']}"
        raw_label = example["label"]
        if isinstance(raw_label, str):
            label_idx = {"A": 0, "B": 1, "C": 2}.get(raw_label, int(raw_label) - 1)
        else:
            label_idx = int(raw_label) - 1
        records.append(
            {
                "instruction": header,
                "input": _build_input(stem, prefixes, choices),
                "output": prefixes[label_idx],
            }
        )
    return records


def build_hellaswag(split: str = "train") -> List[dict]:
    # Use Rowan/hellaswag which has native parquet support (no loading script)
    dataset = load_dataset("Rowan/hellaswag", split=split)
    prefixes = ["ending1", "ending2", "ending3", "ending4"]
    header = "Select the option that best completes the situation (ending1-4)."
    records: List[dict] = []
    for example in dataset:
        context = f"{example['ctx_a']} {example['ctx_b']}".strip()
        records.append(
            {
                "instruction": header,
                "input": _build_input(context, prefixes, list(example["endings"])),
                "output": prefixes[int(example["label"])],
            }
        )
    return records


def build_winogrande(split: str = "train") -> List[dict]:
    # Use allenai/winogrande which has native parquet support (no loading script)
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    prefixes = ["option1", "option2"]
    header = "Choose the word that correctly fills the blank (option1 or option2)."
    records: List[dict] = []
    for example in dataset:
        choices = [example["option1"], example["option2"]]
        label_idx = int(example["answer"]) - 1
        records.append(
            {
                "instruction": header,
                "input": _build_input(example["sentence"], prefixes, choices),
                "output": prefixes[label_idx],
            }
        )
    return records


def _build_arc(config: str, split: str) -> List[dict]:
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    header = "Answer the multiple-choice science question (answer1-5)."
    records: List[dict] = []
    for example in dataset:
        choices = list(example["choices"]["text"])
        labels = list(example["choices"]["label"])
        prefixes = [f"answer{i + 1}" for i in range(len(choices))]
        answer_idx = labels.index(example["answerKey"])
        records.append(
            {
                "instruction": header,
                "input": _build_input(example["question"], prefixes, choices),
                "output": prefixes[answer_idx],
            }
        )
    return records


def build_arc_challenge(split: str = "train") -> List[dict]:
    return _build_arc("ARC-Challenge", split)


def build_arc_easy(split: str = "train") -> List[dict]:
    return _build_arc("ARC-Easy", split)


def build_openbookqa(split: str = "train") -> List[dict]:
    dataset = load_dataset("allenai/openbookqa", "main", split=split)
    header = "Answer the multiple-choice science question (answer1-4)."
    records: List[dict] = []
    for example in dataset:
        choices = example["choices"]["text"]
        prefixes = [f"answer{i + 1}" for i in range(len(choices))]
        facts_list = example.get("facts")
        facts = " ".join(facts_list) if facts_list else ""
        stem = f"{facts}\nQuestion: {example['question_stem']}" if facts else example["question_stem"]
        answer_idx = example["choices"]["label"].index(example["answerKey"])
        records.append(
            {
                "instruction": header,
                "input": _build_input(stem, prefixes, choices),
                "output": prefixes[answer_idx],
            }
        )
    return records


DATASET_REGISTRY: Dict[str, Dict[str, Callable[..., List[dict]]]] = {
    "boolq": {"builder": build_boolq},
    "piqa": {"builder": build_piqa},
    "social_i_qa": {"builder": build_social_iqa},
    "hellaswag": {"builder": build_hellaswag},
    "winogrande": {"builder": build_winogrande},
    "arc_challenge": {"builder": build_arc_challenge},
    "arc_easy": {"builder": build_arc_easy},
    "openbookqa": {"builder": build_openbookqa},
}

ORDERED_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "openbookqa",
]
