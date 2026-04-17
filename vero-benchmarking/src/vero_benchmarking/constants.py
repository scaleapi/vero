from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASETS_DIR = PACKAGE_DIR.parent.parent / "datasets"
DEFAULT_RESULTS_DIR = PACKAGE_DIR.parent.parent / "results"
DEFAULT_LOG_DIR = PACKAGE_DIR.parent.parent / "logs"

for dir_path in [DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DEFAULT_MANIFEST_PATH = DEFAULT_LOG_DIR / "session_manifest.jsonl"
DEFAULT_SEED = 42
