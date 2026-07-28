"""
AFLOW dataset remapping utilities.

This module provides tools to download raw AFLOW datasets and remap them to HuggingFace datasets.
"""

import json
import tarfile
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import gdown
from datasets import Dataset, DatasetDict, load_dataset
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR
from vero_benchmarking.static_data import STATIC_DATA_DIR
from vero_benchmarking.tasks.aflow import AFLOW_TO_HF_DATASETS

AFLOW_RAW_DATASETS_URL = (
    "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
)
AFLOW_RAW_DATASETS_SUBDIR = "aflow"
AFLOW_RAW_DATASET_SPLITS = {"test", "validate"}
AFLOW_TO_HF_MAPPINGS_FILE = "aflow_to_hf_mappings.json"


def download_raw_aflow_datasets(
    url: str = AFLOW_RAW_DATASETS_URL,
    output_dir: Path = DEFAULT_DATASETS_DIR,
    subdir: str = AFLOW_RAW_DATASETS_SUBDIR,
) -> Path:
    """Download and extract the AFLOW datasets tar file from Google Drive.

    Args:
        url: Google Drive download URL.
        output_dir: Parent directory. Defaults to DEFAULT_DATASETS_DIR.
        subdir: Subdirectory name to create and extract into.

    Returns:
        Path to the extracted directory.
    """
    extract_dir = Path(output_dir) / subdir
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "aflow_datasets.tar.gz"
        print(f"Downloading raw AFlow datasets to {tar_path}...")
        gdown.download(url, str(tar_path), quiet=False)

        print(f"Extracting to {extract_dir}...")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir)

    print("Done!")
    return extract_dir


def load_jsonl_file(file_path: Path, dump_values: bool = True) -> list[dict]:
    """Load a jsonl file."""

    def load_line(line: str) -> dict:
        data = json.loads(line)
        if dump_values:
            data_ = {}
            for k in data:
                if isinstance(data[k], (list, dict, int)):
                    data_[k] = json.dumps(data[k])
                else:
                    data_[k] = data[k]
            return data_
        return data

    with open(file_path) as f:
        return [load_line(line) for line in f]


def load_aflow_datasets(
    aflow_dir: Path = DEFAULT_DATASETS_DIR / AFLOW_RAW_DATASETS_SUBDIR,
    splits: set[str] = AFLOW_RAW_DATASET_SPLITS,
) -> dict[str, DatasetDict]:
    """Extract dataset names and their splits from the aflow directory.

    Args:
        aflow_dir: Path to the aflow datasets directory.

    Returns:
        Dictionary mapping dataset names to DatasetDicts.
    """
    datasets = defaultdict(dict)

    for file in Path(aflow_dir).glob("*.jsonl"):
        stem = file.stem
        dataset_name, split = stem.split("_", 1)
        if split not in splits:
            continue

        datasets[dataset_name][split] = Dataset.from_list(load_jsonl_file(file))

    for dataset_name in datasets:
        datasets[dataset_name] = DatasetDict(datasets[dataset_name])

    return dict(datasets)


def load_hf_datasets(dataset_paths: dict[str, str | tuple[str, str]]) -> dict[str, DatasetDict]:
    """Load HF datasets from the given paths.

    Args:
        dataset_paths: Dictionary mapping dataset names to paths.

    Returns:
        Dictionary mapping dataset names to DatasetDicts.
    """
    datasets = {}
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"Loading {dataset_path}...")
        if isinstance(dataset_path, tuple):
            dataset_path, dataset_name = dataset_path
            datasets[dataset_path] = load_dataset(dataset_path, dataset_name)
        else:
            datasets[dataset_path] = load_dataset(dataset_path)
    return datasets


def build_index_for_dataset(
    dataset_dict: DatasetDict, field_name: str | int | Callable, unique: bool = True
) -> dict[str | int, tuple[str, int]]:
    """Build an index for a dataset.

    Args:
        dataset_dict: Dataset dictionary.
        field_name: Field name to index on.
        unique: Whether the key should be unique.

    Returns:
        Dictionary mapping keys to (split, index).
    """
    id_to_sample = {}
    for split in dataset_dict.keys():
        for idx, example in enumerate(dataset_dict[split]):
            if not isinstance(field_name, (str, int)):
                key = field_name(example)
            else:
                key = example[field_name]

            if key in id_to_sample and unique:
                raise ValueError(f"Key {key} already exists. Not unique.")

            id_to_sample[key] = (split, idx)
    return id_to_sample


def get_aflow_mapping(
    aflow_dataset_dict: DatasetDict,
    hf_dataset_dict: DatasetDict,
    field_name: str,
    key_type: type = str,
) -> dict:
    """Get the mapping between AFLOW and HF datasets using a given field name.

    Args:
        field_name: Field name to use for the mapping.
        aflow_dataset_dict: AFLOW dataset dictionary.
        hf_dataset_dict: HF dataset dictionary.
    """

    id_to_sample = build_index_for_dataset(hf_dataset_dict, field_name)
    utilized_keys = set()
    mapping = defaultdict(list)

    for aflow_split in aflow_dataset_dict.keys():
        for aflow_example in aflow_dataset_dict[aflow_split]:
            key = key_type(aflow_example[field_name])

            if key not in id_to_sample:
                raise ValueError(f"Key {key} not found in HF datasets.")

            if key in utilized_keys:
                raise ValueError(f"Key {key} was already utilized by another example.")

            utilized_keys.add(key)
            mapping[aflow_split].append(id_to_sample[key])

    return mapping


get_mbpp_mapping = partial(get_aflow_mapping, field_name="task_id", key_type=int)
get_gsm8k_mapping = partial(get_aflow_mapping, field_name="question")
get_math_mapping = partial(get_aflow_mapping, field_name="problem")
get_humaneval_mapping = partial(get_aflow_mapping, field_name="task_id")


def get_hotpotqa_mapping(aflow_dataset_dict: DatasetDict, hf_dataset_dict: DatasetDict) -> dict:
    """Get the mapping between AFLOW and HF HotpotQA datasets using the `id` (HF) and `_id` (AFlow) fields.

    Args:
        aflow_dataset_dict: AFLOW dataset dictionary.
        hf_dataset_dict: HF dataset dictionary.
    """
    hf_field_name = "id"
    aflow_field_name = "_id"
    id_to_sample = build_index_for_dataset(hf_dataset_dict, hf_field_name)

    def best_guess_key(example: dict) -> str:
        return (example["question"], example["answer"], example["level"])

    best_guess_key_to_sample = build_index_for_dataset(
        hf_dataset_dict, best_guess_key, unique=False
    )

    utilized_keys = set()
    mapping = defaultdict(list)

    missing_keys = dict()

    for aflow_split in aflow_dataset_dict.keys():
        for aflow_example in aflow_dataset_dict[aflow_split]:
            key = aflow_example[aflow_field_name]

            if key in utilized_keys:
                raise ValueError(f"Key {key} was already utilized by another example.")

            if key not in id_to_sample:
                key = best_guess_key(aflow_example)

                if key not in best_guess_key_to_sample:
                    raise ValueError(f"Key {key} not found in HF datasets.")

                if key in utilized_keys:
                    raise ValueError(f"Key {key} was already utilized by another example.")

                utilized_keys.add(key)
                mapping[aflow_split].append(best_guess_key_to_sample[key])
                continue

            utilized_keys.add(key)
            mapping[aflow_split].append(id_to_sample[key])

    if missing_keys:
        print("Could not find the following questions in HF datasets:")
        for key, example in missing_keys.items():
            print(example["question"])
        raise ValueError(f"Keys {missing_keys} not found in HF datasets.")

    return mapping


def get_drop_mapping(aflow_dataset_dict: DatasetDict, hf_dataset_dict: DatasetDict) -> dict:
    """Get the mapping between AFLOW and HF DROP datasets using the `passage` and `question` fields.

    Args:
        aflow_dataset_dict: AFLOW dataset dictionary.
        hf_dataset_dict: HF dataset dictionary.
    """

    def get_hf_key(example: dict) -> tuple[str, str]:
        passage = example["passage"].strip()
        question = example["question"].strip()
        return passage, question

    id_to_sample = build_index_for_dataset(hf_dataset_dict, get_hf_key, unique=False)
    utilized_keys = set()
    mapping = defaultdict(list)

    def extract_passage_question(context: str) -> tuple[str, str]:
        """Extract the passage and question from the context.
        Format is 'Passage: ....\nQuestion: ....\nAnswer: ....'
        """
        context = context.split("Answer:")[0].strip()
        passage, question = context.split("Question:")
        question = question.strip()
        passage = passage.split("Passage:")[1].strip()
        return passage, question

    for aflow_split in aflow_dataset_dict.keys():
        for aflow_example in aflow_dataset_dict[aflow_split]:
            context = aflow_example["context"]
            passage, question = extract_passage_question(context)
            key = (passage, question)

            if key not in id_to_sample:
                raise ValueError(f"Key {key} not found in HF datasets.")

            utilized_keys.add(key)
            mapping[aflow_split].append(id_to_sample[key])

    return mapping


MAPPING_FUNCTIONS = {
    "mbpp": get_mbpp_mapping,
    "gsm8k": get_gsm8k_mapping,
    "math": get_math_mapping,
    "hotpotqa": get_hotpotqa_mapping,
    "humaneval": get_humaneval_mapping,
    "drop": get_drop_mapping,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading the raw AFlow datasets."
    )
    parser.add_argument(
        "--skip-remap", action="store_true", help="Skip remapping the AFlow datasets."
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_raw_aflow_datasets()

    if args.skip_remap:
        exit()

    aflow_datasets = load_aflow_datasets()

    print("Loading HF datasets...")
    hf_datasets = load_hf_datasets(AFLOW_TO_HF_DATASETS)
    print("Done!")

    mappings = {}

    for dataset_name, mapping_function in MAPPING_FUNCTIONS.items():
        hf_dataset_path = AFLOW_TO_HF_DATASETS[dataset_name]
        if isinstance(hf_dataset_path, tuple):
            hf_dataset_path = hf_dataset_path[0]

        print(f"Mapping {dataset_name} to {hf_dataset_path}...")
        mapping = mapping_function(aflow_datasets[dataset_name], hf_datasets[hf_dataset_path])
        mappings[dataset_name] = mapping

    # Dump to JSON in static_data folder
    with open(STATIC_DATA_DIR / AFLOW_TO_HF_MAPPINGS_FILE, "w") as f:
        json.dump(mappings, f, indent=4)

    print("Done!")
