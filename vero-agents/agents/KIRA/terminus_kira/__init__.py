import os

# Use public Harbor registry for terminal-bench datasets (must be set before harbor import)
os.environ.setdefault("INTERNAL_REGISTRY", "0")

from terminus_kira.terminus_kira import TerminusKira

__all__ = ["TerminusKira"]
