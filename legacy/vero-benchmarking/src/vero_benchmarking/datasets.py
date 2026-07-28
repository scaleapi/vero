import json
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import kagglehub
import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter

from datasets import Dataset, DatasetDict, load_dataset
from vero_benchmarking.constants import DEFAULT_DATASETS_DIR, DEFAULT_SEED
from vero_benchmarking.static_data import (
    AFLOW_TO_HF_MAPPINGS,
    GAIA_PURE_LANGUAGE_MAPPINGS,
    SIMPLEQA_GPT41_MINI_UNANSWERED_INDICES,
    SIMPLEQA_UNANSWERED_INDICES,
    TAU_BENCH_RETAIL_RESULTS,
)
from vero_benchmarking.tasks.aflow import AFLOW_TO_HF_DATASETS


class DatasetBuilder(BaseModel, ABC):
    """An abstract base class for consolidating the logic for building datasets from raw sources."""

    name: Any

    @abstractmethod
    def build(self) -> DatasetDict | Dataset: ...

    def build_and_save(self, path: str) -> DatasetDict | Dataset:
        """Build the dataset and save it to a given path."""
        ds = self.build()
        ds.save_to_disk(path)
        return ds


class TauBenchRetail(DatasetBuilder):
    name: Literal["tau_bench_retail"] = "tau_bench_retail"
    train_sample: float = 100
    validation_sample: float | None = None
    test_sample: float | None = None
    seed: int = DEFAULT_SEED

    def build(self) -> DatasetDict | Dataset:
        df = pd.read_csv(TAU_BENCH_RETAIL_RESULTS)

        initial_columns = df.columns

        df["task_id"] = df["dataset_sample_sample_id"]
        df["task_split"] = df["split"]

        df.drop(initial_columns, axis=1, inplace=True)

        val_df = df[df.task_split == "validation"]
        test_df = df[df.task_split == "test"]
        train_df = df[df.task_split == "train"]
        train_df = train_df.sample(self.train_sample, random_state=self.seed)

        train_ds = Dataset.from_pandas(train_df)
        validation_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)

        return DatasetDict(
            {"train": train_ds, "validation": validation_ds, "test": test_ds}
        )


class GaiaPureLanguage(DatasetBuilder):
    """GAIA benchmark filtered to pure-language questions, with pre-computed split mappings."""

    name: Literal["gaia_pure_language"] = "gaia_pure_language"
    mappings_file: Path = GAIA_PURE_LANGUAGE_MAPPINGS

    def build(self) -> DatasetDict:
        hf_ds = load_dataset("gaia-benchmark/GAIA", "2023_all")
        mappings = json.loads(self.mappings_file.read_text())

        # Exclude test split — GAIA test answers are hidden (all '?')
        splits = {}
        for split_name, indices in mappings.items():
            if split_name == "test":
                continue
            samples = [hf_ds[hf_split][hf_idx] for hf_split, hf_idx in indices]
            splits[split_name] = Dataset.from_list(samples)

        return DatasetDict(splits)


class GPQADiamond(DatasetBuilder):
    name: Literal["gpqa_diamond"] = "gpqa_diamond"
    test_size: float = 100
    validation_size: float = 0.5
    seed: int = DEFAULT_SEED

    def _load_and_transform(self) -> Dataset:
        """Load and transform GPQA diamond dataset with shuffled answer options."""
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        ds = ds["train"]

        # assign each example a correct answer index such that the distribution of answer indices (over 0, 1, 2, 3) is uniform
        num_samples = len(ds)
        num_samples_per_answer_index = math.ceil(num_samples / 4)
        index_assignments = sum(
            [[i] * num_samples_per_answer_index for i in range(4)], []
        )
        index_assignments = index_assignments[:num_samples]

        rng = random.Random(self.seed)
        rng.shuffle(index_assignments)

        def construct_example(example: dict, idx: int) -> dict:
            rng = random.Random(self.seed + idx)
            options = [
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
            rng.shuffle(options)
            correct_idx = index_assignments[idx]
            options.insert(correct_idx, example["Correct Answer"])

            example_ = {
                "question": example["Question"],
                "options": options,
                "explanation": example["Explanation"],
                "answer": example["Correct Answer"],
                "answer_index": correct_idx,
            }
            return example_

        return ds.map(
            construct_example, with_indices=True, remove_columns=ds.column_names
        )

    def build(self) -> DatasetDict | Dataset:
        """Build GPQA diamond with train, validation, and test splits."""
        ds = self._load_and_transform()
        ds = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        test_split = ds["test"]
        train_validation_split = ds["train"].train_test_split(
            test_size=self.validation_size, seed=self.seed
        )
        train_split = train_validation_split["train"]
        validation_split = train_validation_split["test"]
        return DatasetDict(
            {"train": train_split, "validation": validation_split, "test": test_split}
        )


class GPQADiamondNoSplit(GPQADiamond):
    """GPQA Diamond without validation split - only train and test."""

    name: Literal["gpqa_diamond_no_split"] = "gpqa_diamond_no_split"

    def build(self) -> DatasetDict | Dataset:
        """Build GPQA diamond with only train and test splits (no validation)."""
        ds = self._load_and_transform()
        ds = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        return DatasetDict({"train": ds["train"], "test": ds["test"]})


class FactsSearch(DatasetBuilder):
    name: Literal["facts_search"] = "facts_search"
    test_size: float = 0.8
    validation_size: float = 0.5
    seed: int = DEFAULT_SEED

    def build(self) -> DatasetDict | Dataset:
        path = kagglehub.dataset_download("deepmind/facts-search-public")
        path = Path(path) / "facts_open_filtered.csv"
        df = pd.read_csv(path)
        ds = Dataset.from_pandas(df)
        ds = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        test_split = ds["test"]
        train_validation_split = ds["train"].train_test_split(
            test_size=self.validation_size, seed=self.seed
        )
        train_split = train_validation_split["train"]
        validation_split = train_validation_split["test"]
        return DatasetDict(
            {"train": train_split, "validation": validation_split, "test": test_split}
        )


class SimpleQAVerifiedWikiOnly(DatasetBuilder):
    name: Literal["simple_qa_verified_wiki_only"] = "simple_qa_verified_wiki_only"

    @staticmethod
    def answer_on_wikipedia(sample: dict) -> bool:
        """Best-guess heuristic to filter out samples that cannot be answered with information on Wikipedia."""
        sample["urls"]: str
        return any("wikipedia" in url for url in sample["urls"].split(","))

    def build(self) -> DatasetDict | Dataset:
        """Load the original SimpleQA-Verified dataset and filter out samples that cannot be answered with information on Wikipedia."""
        ds = load_dataset("google/simpleqa-verified")
        ds = ds.filter(self.answer_on_wikipedia)
        return ds["eval"]


class SimpleQAVerifiedWikiUnanswered(SimpleQAVerifiedWikiOnly):
    """A subset of Simple-QA Verified Wiki Only that only includes questions that cannot be answered by base models without tools."""

    name: Literal["simple_qa_verified_wiki_unanswered"] = (
        "simple_qa_verified_wiki_unanswered"
    )
    path_to_filter_idxs: Path = SIMPLEQA_UNANSWERED_INDICES
    test_size: int = 80
    validation_size: int = 45
    seed: int = DEFAULT_SEED

    def build(self) -> DatasetDict:
        ds = super().build()
        filter_idxs: list[int] = json.loads(self.path_to_filter_idxs.read_text())
        ds = ds.select(filter_idxs)
        ds = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        test_split = ds["test"]
        train_validation_split = ds["train"].train_test_split(
            test_size=self.validation_size, seed=self.seed
        )
        train_split = train_validation_split["train"]
        validation_split = train_validation_split["test"]
        return DatasetDict(
            {"train": train_split, "validation": validation_split, "test": test_split}
        )


class SimpleQAVerifiedWikiGPT41MiniUnanswered(SimpleQAVerifiedWikiOnly):
    """A subset of Simple-QA Verified Wiki Only that only includes questions that cannot be answered by GPT-4.1 Mini without tools."""

    name: Literal["simple_qa_verified_wiki_gpt41_mini_unanswered"] = (
        "simple_qa_verified_wiki_gpt41_mini_unanswered"
    )
    path_to_filter_idxs: Path = SIMPLEQA_GPT41_MINI_UNANSWERED_INDICES
    test_size: int = 100
    validation_size: int = 100
    train_size: int = 100
    seed: int = DEFAULT_SEED

    def build(self) -> DatasetDict:
        ds = super().build()
        filter_idxs: list[int] = json.loads(self.path_to_filter_idxs.read_text())
        ds = ds.select(filter_idxs)
        train_size = self.train_size + self.validation_size
        ds = ds.train_test_split(
            train_size=train_size, test_size=self.test_size, seed=self.seed
        )
        test_split = ds["test"]
        train_validation_split = ds["train"].train_test_split(
            test_size=self.validation_size, seed=self.seed
        )
        train_split = train_validation_split["train"]
        validation_split = train_validation_split["test"]
        return DatasetDict(
            {"train": train_split, "validation": validation_split, "test": test_split}
        )


class AflowDatasetBuilder(DatasetBuilder):
    """Base class for building AFLOW datasets from HuggingFace sources using pre-computed mappings."""

    aflow_dataset_name: str
    mappings_file: Path = AFLOW_TO_HF_MAPPINGS
    train_validation_split_name: str = "validate"
    train_size: float = 0.5
    test_size: float | None = None
    seed: int = DEFAULT_SEED
    skip_validation_split: bool = False

    def _load_hf_dataset(self) -> DatasetDict:
        hf_path = AFLOW_TO_HF_DATASETS[self.aflow_dataset_name]
        if isinstance(hf_path, tuple):
            return load_dataset(hf_path[0], hf_path[1])
        return load_dataset(hf_path)

    def _load_mappings(self) -> dict[str, list[tuple[str, int]]]:
        all_mappings = json.loads(self.mappings_file.read_text())
        return all_mappings[self.aflow_dataset_name]

    def build_remapped(self) -> DatasetDict:
        hf_ds = self._load_hf_dataset()
        mappings = self._load_mappings()

        splits = {}
        for aflow_split, indices in mappings.items():
            samples = [hf_ds[hf_split][hf_idx] for hf_split, hf_idx in indices]
            splits[aflow_split] = Dataset.from_list(samples)

        return DatasetDict(splits)

    def build(self) -> DatasetDict:
        ds_dict = self.build_remapped()
        train_validation_ds = ds_dict.pop(self.train_validation_split_name)

        if self.skip_validation_split:
            ds_dict["train"] = train_validation_ds
        else:
            train_validation_ds_dict = train_validation_ds.train_test_split(
                train_size=self.train_size, test_size=self.test_size, seed=self.seed
            )
            ds_dict["train"] = train_validation_ds_dict["train"]
            ds_dict["validation"] = train_validation_ds_dict["test"]

        return ds_dict


class AflowDrop(AflowDatasetBuilder):
    name: Literal["aflow_drop"] = "aflow_drop"
    aflow_dataset_name: str = "drop"


class AflowDropSingleAnswer(AflowDrop):
    name: Literal["aflow_drop_single_answer"] = "aflow_drop_single_answer"
    aflow_dataset_name: str = "drop"

    def build(self) -> DatasetDict:
        ds_dict = super().build()

        # Filter out examples with multiple answer spans
        def filter_single_answer(example: dict) -> bool:
            return len(set(example["answers_spans"]["types"])) == 1

        def get_single_answer(example: dict) -> str:
            example["answer"] = example["answers_spans"]["spans"][0]
            return example

        ds_dict = ds_dict.filter(filter_single_answer)
        ds_dict = ds_dict.map(get_single_answer)
        return ds_dict


class AflowDropSingleAnswerNoSplit(AflowDropSingleAnswer):
    name: Literal["aflow_drop_single_answer_no_split"] = (
        "aflow_drop_single_answer_no_split"
    )
    skip_validation_split: bool = True


class AflowGsm8k(AflowDatasetBuilder):
    name: Literal["aflow_gsm8k"] = "aflow_gsm8k"
    aflow_dataset_name: str = "gsm8k"


class AflowGsm8kNoSplit(AflowDatasetBuilder):
    name: Literal["aflow_gsm8k_no_split"] = "aflow_gsm8k_no_split"
    aflow_dataset_name: str = "gsm8k"
    skip_validation_split: bool = True


class AflowHotpotqa(AflowDatasetBuilder):
    name: Literal["aflow_hotpotqa"] = "aflow_hotpotqa"
    aflow_dataset_name: str = "hotpotqa"


class AflowHotpotqaNoSplit(AflowDatasetBuilder):
    name: Literal["aflow_hotpotqa_no_split"] = "aflow_hotpotqa_no_split"
    aflow_dataset_name: str = "hotpotqa"
    skip_validation_split: bool = True


class AflowHumaneval(AflowDatasetBuilder):
    name: Literal["aflow_humaneval"] = "aflow_humaneval"
    aflow_dataset_name: str = "humaneval"


class AflowHumanevalNoSplit(AflowDatasetBuilder):
    name: Literal["aflow_humaneval_no_split"] = "aflow_humaneval_no_split"
    aflow_dataset_name: str = "humaneval"
    skip_validation_split: bool = True


class AflowMath(AflowDatasetBuilder):
    name: Literal["aflow_math"] = "aflow_math"
    aflow_dataset_name: str = "math"


class AflowMathNoSplit(AflowDatasetBuilder):
    name: Literal["aflow_math_no_split"] = "aflow_math_no_split"
    aflow_dataset_name: str = "math"
    skip_validation_split: bool = True


class AflowMbpp(AflowDatasetBuilder):
    name: Literal["aflow_mbpp"] = "aflow_mbpp"
    aflow_dataset_name: str = "mbpp"


class AflowMbppNoSplit(AflowDatasetBuilder):
    name: Literal["aflow_mbpp_no_split"] = "aflow_mbpp_no_split"
    aflow_dataset_name: str = "mbpp"
    skip_validation_split: bool = True


DatasetBuilderT = Annotated[
    Union[
        FactsSearch,
        SimpleQAVerifiedWikiUnanswered,
        SimpleQAVerifiedWikiGPT41MiniUnanswered,
        AflowDrop,
        AflowDropSingleAnswer,
        AflowDropSingleAnswerNoSplit,
        AflowGsm8k,
        AflowGsm8kNoSplit,
        AflowHotpotqa,
        AflowHotpotqaNoSplit,
        AflowHumaneval,
        AflowHumanevalNoSplit,
        AflowMath,
        AflowMathNoSplit,
        AflowMbpp,
        AflowMbppNoSplit,
        GaiaPureLanguage,
        GPQADiamond,
        GPQADiamondNoSplit,
        TauBenchRetail,
    ],
    Field(discriminator="name"),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False, default=None)
    parser.add_argument("--no-mkdir", action="store_false")
    args = parser.parse_args()

    dataset_builder = TypeAdapter(DatasetBuilderT).validate_python(
        {"name": args.dataset_name}
    )

    if args.output_path is None:
        args.output_path = DEFAULT_DATASETS_DIR / dataset_builder.name

    if not args.no_mkdir:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    ds = dataset_builder.build_and_save(args.output_path)
