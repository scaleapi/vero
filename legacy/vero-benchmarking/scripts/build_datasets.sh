#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Benchmark datasets (required for paper experiments)
BENCHMARK_DATASETS=(
    "aflow_math"
    "gpqa_diamond_no_split"
    "simple_qa_verified_wiki_unanswered"
    "tau_bench_retail"
    "gaia_pure_language"
)

# Additional AFLOW datasets
AFLOW_DATASETS=(
    "aflow_drop"
    "aflow_drop_single_answer"
    "aflow_drop_single_answer_no_split"
    "aflow_gsm8k"
    "aflow_gsm8k_no_split"
    "aflow_hotpotqa"
    "aflow_hotpotqa_no_split"
    "aflow_humaneval"
    "aflow_humaneval_no_split"
    "aflow_math_no_split"
    "aflow_mbpp"
    "aflow_mbpp_no_split"
)

# Other datasets
OTHER_DATASETS=(
    "simple_qa_verified_wiki_gpt41_mini_unanswered"
    "facts_search"
    "gpqa_diamond"
)

if [ "$1" = "--all" ]; then
    DATASETS=("${BENCHMARK_DATASETS[@]}" "${AFLOW_DATASETS[@]}" "${OTHER_DATASETS[@]}")
else
    DATASETS=("${BENCHMARK_DATASETS[@]}")
fi

echo "Building ${#DATASETS[@]} datasets..."
echo

for dataset in "${DATASETS[@]}"; do
    echo "Building $dataset..."
    python -m vero_benchmarking.datasets --dataset-name "$dataset"
    echo "Done building $dataset"
    echo
done

echo "All datasets built successfully!"
