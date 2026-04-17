from pathlib import Path

CORE_PACKAGE_DIR = Path(__file__).resolve().parent.resolve()
PACKAGE_DIR = CORE_PACKAGE_DIR.parent.parent.parent.resolve()
SCAFFOLDS_DIR = CORE_PACKAGE_DIR / "scaffolds"
VEROACCESS_FILENAME = ".veroaccess"
_DEFAULT_VEROACCESS_PATH = SCAFFOLDS_DIR / "default.veroaccess"


evaluation_results_basename = "evaluation_results.json"
evaluation_parameters_basename = "evaluation_parameters.json"
result_metadata_basename = "result_metadata.json"
pytest_report_basename = "pytest_report.json"
samples_dir_name = "samples"

default_minimum_score = 0.0
default_maximum_score = 1.0

context_artifacts_directory = Path(__file__).parent.parent / "skills"
context_artifacts_namespace = "agent-cookbooks"
