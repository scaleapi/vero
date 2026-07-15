import json

import pytest

from vero_benchmarking.cases import materialize_cases


def test_materialize_jsonl_adds_ids_and_freezes_session_input(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"question": "one"}\n{"id": "two", "question": "two"}\n')
    output = tmp_path / "session" / "cases.jsonl"

    assert materialize_cases(
        dataset_path=source,
        partition="test",
        output_path=output,
    ) == output.resolve()
    cases = [json.loads(line) for line in output.read_text().splitlines()]
    assert [case["id"] for case in cases] == ["0", "two"]

    materialize_cases(dataset_path=source, partition="test", output_path=output)
    source.write_text('{"question": "changed"}\n')
    with pytest.raises(ValueError, match="existing session changed"):
        materialize_cases(dataset_path=source, partition="test", output_path=output)


def test_materialize_rejects_duplicate_stringified_ids(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"id": 1}, {"id": "1"}]))

    with pytest.raises(ValueError, match="case IDs must be unique"):
        materialize_cases(
            dataset_path=source,
            partition="test",
            output_path=tmp_path / "cases.jsonl",
        )


def test_materialize_dataset_dict_selects_partition(tmp_path):
    from datasets import Dataset, DatasetDict

    source = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list([{"value": "train"}]),
            "test": Dataset.from_list([{"value": "test"}]),
        }
    ).save_to_disk(source)

    output = materialize_cases(
        dataset_path=source,
        partition="test",
        output_path=tmp_path / "cases.jsonl",
    )
    assert json.loads(output.read_text()) == {"value": "test", "id": "0"}
