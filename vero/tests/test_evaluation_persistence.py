import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from vero.core.db.candidate import Candidate
from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import (
    ExperimentResult,
    ExperimentResultStatus,
    SampleResult,
)
from vero.core.db.run import ExperimentRun
from vero.core.evaluation import EvaluationParameters
from vero.evaluation import (
    BackendProvenance,
    CaseCheckpointStore,
    CaseError,
    CaseResult,
    CaseStatus,
    EvaluationArtifact,
    EvaluationDatabase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    EvaluationStore,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)


def _record(
    *,
    commit: str = "abc",
    value: float = 1.0,
    feasible: bool = True,
    record_id: str | None = None,
) -> EvaluationRecord:
    created_at = datetime(2026, 1, 1)
    spec = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
        failure_value=0.0,
    )
    return EvaluationRecord(
        id=record_id or str(uuid4()),
        request=EvaluationRequest(
            candidate=Candidate(
                commit=commit,
                repo_name="repo",
                created_at=created_at,
            ),
            evaluation_set=EvaluationSet(name="validation"),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": value},
            cases=[
                CaseResult(
                    case_id="case/one",
                    status=CaseStatus.SUCCESS,
                    metrics={"score": value},
                    artifacts=[EvaluationArtifact(path="cases/one.log")],
                ),
                CaseResult(
                    case_id="case two",
                    status=CaseStatus.ERROR,
                    errors=[CaseError(message="failed", terminal=True)],
                ),
            ],
            artifacts=[EvaluationArtifact(path="summary.json")],
        ),
        backend_id="default",
        backend=BackendProvenance(
            name="fake",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=spec,
        objective=ObjectiveResult(value=value, feasible=feasible),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


@pytest.mark.asyncio
async def test_checkpoint_store_hashes_case_ids_and_round_trips(tmp_path: Path):
    store = CaseCheckpointStore(tmp_path / "cases")
    result = CaseResult(
        case_id="../../not-a-filename",
        status=CaseStatus.SUCCESS,
        metrics={"score": 1.0},
    )

    await store.save(result)

    paths = list((tmp_path / "cases").iterdir())
    assert len(paths) == 1
    assert paths[0].name.endswith(".json")
    assert result.case_id not in paths[0].name
    assert await store.load(result.case_id) == result


@pytest.mark.asyncio
async def test_checkpoint_store_reports_corruption_with_path(tmp_path: Path):
    store = CaseCheckpointStore(tmp_path / "cases")
    case_id = "case"
    path = store.path_for(case_id)
    path.parent.mkdir(parents=True)
    path.write_text("not-json")

    with pytest.raises(ValueError, match=str(path)):
        await store.load(case_id)


@pytest.mark.asyncio
async def test_evaluation_store_writes_cases_then_manifest_and_round_trips(
    tmp_path: Path,
):
    record = _record()
    store = EvaluationStore(tmp_path / record.id)

    await store.save(record)

    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["lifecycle"] == "complete"
    assert manifest["report"]["cases"] == []
    assert [case["case_id"] for case in manifest["case_files"]] == [
        "case/one",
        "case two",
    ]
    for case_file in manifest["case_files"]:
        assert (store.result_dir / case_file["path"]).exists()
    assert store.load() == record


@pytest.mark.asyncio
async def test_manifest_replace_failure_never_exposes_complete_manifest(
    tmp_path: Path, monkeypatch
):
    from vero.evaluation import persistence

    record = _record()
    store = EvaluationStore(tmp_path / record.id)
    original_replace = persistence.os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination) == store.manifest_path:
            raise OSError("simulated interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        await store.save(record)

    assert not store.manifest_path.exists()
    assert list(store.result_dir.glob(".evaluation.json.*.tmp")) == []


@pytest.mark.asyncio
async def test_corrupt_case_file_names_the_affected_path(tmp_path: Path):
    record = _record()
    store = EvaluationStore(tmp_path / record.id)
    await store.save(record)
    manifest = json.loads(store.manifest_path.read_text())
    corrupt_path = store.result_dir / manifest["case_files"][0]["path"]
    corrupt_path.write_text("{")

    with pytest.raises(ValueError, match=str(corrupt_path)):
        store.load()


@pytest.mark.asyncio
async def test_artifact_symlink_cannot_escape_artifact_root(tmp_path: Path):
    record = _record()
    store = EvaluationStore(tmp_path / record.id)
    store.artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.artifact_dir / "cases").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        await store.save(record)


def test_evaluation_database_round_trips_schema_v2_and_selects_best(tmp_path: Path):
    database = EvaluationDatabase(id="session")
    slower = _record(commit="slow", value=0.5)
    faster = _record(commit="fast", value=1.0)
    infeasible = _record(commit="invalid", value=100.0, feasible=False)
    for record in (slower, faster, infeasible):
        database.add_evaluation(record)

    path = tmp_path / "database.json"
    database.save_to_file(path)
    restored = EvaluationDatabase.load_from_file(path)

    assert json.loads(path.read_text())["schema_version"] == 2
    assert restored.serialize() == database.serialize()
    assert restored.get_best(faster.objective_spec) == faster
    assert (
        restored.get_best(
            faster.objective_spec,
            exclude_candidate=faster.request.candidate,
        )
        == slower
    )

    frame = restored.get_evaluations_df([faster])
    assert frame.index.tolist() == [faster.id]
    assert frame.loc[faster.id, "candidate_commit"] == "fast"
    assert frame.loc[faster.id, "objective_value"] == 1.0
    assert frame.loc[faster.id, "metric/score"] == 1.0


def test_evaluation_database_distinguishes_no_id_filter_from_empty_id_filter():
    database = EvaluationDatabase(id="session")
    record = _record()
    database.add_evaluation(record)

    assert database.get_evaluations() == [record]
    assert database.get_evaluations([]) == []


def test_database_concurrent_inserts_and_atomic_saves_do_not_lose_records(
    tmp_path: Path,
):
    database = EvaluationDatabase(id="session")
    records = [_record(commit=f"c-{index}") for index in range(20)]
    path = tmp_path / "database.json"

    def insert_and_save(record):
        database.add_evaluation(record)
        database.save_to_file(path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(insert_and_save, records))
    database.save_to_file(path)

    restored = EvaluationDatabase.load_from_file(path)
    assert set(restored.evaluations) == {record.id for record in records}


@pytest.mark.asyncio
async def test_database_reconstructs_evaluations_and_skips_only_corrupt_record(
    tmp_path: Path, caplog
):
    experiments_dir = tmp_path / "experiments"
    good = _record(commit="good")
    await EvaluationStore(experiments_dir / good.id).save(good)
    corrupt_dir = experiments_dir / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "evaluation.json").write_text("not-json")

    database = EvaluationDatabase.from_evaluations_dir(
        experiments_dir,
        db_id="session",
    )

    assert list(database.evaluations.values()) == [good]
    assert "Skipping corrupt evaluation" in caplog.text


def test_database_reconstructs_schema_v1_result_directories(tmp_path: Path):
    experiments_dir = tmp_path / "experiments"
    result_dir = experiments_dir / "legacy-result"
    samples_dir = result_dir / "samples"
    samples_dir.mkdir(parents=True)
    candidate = Candidate(commit="legacy", repo_name="repo")
    run = ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(
            dataset_id="benchmark",
            split="validation",
            sample_ids=[3],
        ),
    )
    parameters = EvaluationParameters(
        result_id="legacy-result",
        run=run,
        session_id="legacy-session",
    )
    (result_dir / "evaluation_parameters.json").write_text(
        parameters.model_dump_json(indent=2)
    )
    sample = SampleResult(
        dataset_sample=DatasetSample(
            dataset_id="benchmark",
            split="validation",
            sample_id=3,
        ),
        score=0.75,
    )
    (samples_dir / "3.json").write_text(sample.model_dump_json(indent=2))
    (result_dir / "result_metadata.json").write_text(
        json.dumps(
            {
                "id": "legacy-result",
                "run_id": run.id,
                "status": "success",
            }
        )
    )

    database = EvaluationDatabase.from_evaluations_dir(
        experiments_dir,
        db_id="reconstructed",
    )

    record = database.get_evaluation("legacy-result")
    assert record is not None
    assert record.backend_id == "vero-task"
    assert record.report.cases[0].case_id == "3"
    assert record.objective.value == 0.75


def test_schema_v1_database_converts_to_canonical_records_without_rewrite(
    tmp_path: Path,
):
    candidate = Candidate(
        commit="legacy-commit",
        repo_name="legacy-repo",
        created_at=datetime(2025, 1, 1),
    )
    run = ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(
            dataset_id="benchmark",
            split="validation",
            sample_ids=[2, 7],
        ),
    )
    result = ExperimentResult(
        id="legacy-result",
        run_id=run.id,
        status=ExperimentResultStatus.SUCCESS,
        sample_results={
            2: SampleResult(
                dataset_sample=DatasetSample(
                    dataset_id="benchmark",
                    split="validation",
                    sample_id=2,
                ),
                score=1.0,
                metrics={"score": 99.0, "custom": 2.0},
                input={"question": "private"},
                output="answer",
            ),
            7: SampleResult(
                dataset_sample=DatasetSample(
                    dataset_id="benchmark",
                    split="validation",
                    sample_id=7,
                ),
                error="execution failed",
                eval_error="scoring also failed",
            ),
        },
    )
    legacy = ExperimentDatabase(id="legacy-session")
    legacy.add_experiment(Experiment(run=run, result=result))
    source = tmp_path / "legacy.json"
    source.write_text(legacy.to_json())
    original_source = source.read_text()

    converted = EvaluationDatabase.load_from_file(source)
    record = converted.get_evaluation("legacy-result")

    assert record.backend_id == "vero-task"
    assert record.backend.version == "legacy-v1"
    assert record.request.evaluation_set.name == "benchmark"
    assert record.request.evaluation_set.partition == "validation"
    assert record.request.evaluation_set.selection.ids == ["2", "7"]
    assert record.report.cases[0].metrics == {"score": 1.0, "custom": 2.0}
    assert record.report.diagnostics[0].code == "legacy_score_metric_collision"
    assert [error.phase for error in record.report.cases[1].errors] == [
        "execution",
        "scoring",
    ]
    assert record.report.cases[1].errors[-1].terminal is True
    assert record.objective.value == 0.5
    assert record.objective.feasible is True

    destination = tmp_path / "canonical.json"
    converted.save_to_file(destination)
    assert source.read_text() == original_source
    assert json.loads(destination.read_text())["schema_version"] == 2


def test_legacy_failure_without_message_gets_structured_terminal_error():
    candidate = Candidate(commit="c", repo_name="repo")
    run = ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(dataset_id="data", split="test"),
    )
    result = ExperimentResult(
        id="result",
        run_id=run.id,
        status=ExperimentResultStatus.FAILED,
        sample_results={
            0: SampleResult(
                dataset_sample=DatasetSample(
                    dataset_id="data", split="test", sample_id=0
                )
            )
        },
    )
    legacy = ExperimentDatabase(id="session")
    legacy.add_experiment(Experiment(run=run, result=result))

    converted = EvaluationDatabase.deserialize(legacy.serialize())
    case = converted.get_evaluation("result").report.cases[0]

    assert case.status == CaseStatus.ERROR
    assert case.errors[0].code == "legacy_missing_error"
    assert case.errors[0].terminal is True
