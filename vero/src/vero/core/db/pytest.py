from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pandas import Series


class PyTestExitCode(IntEnum):
    """Exit codes for pytest execution."""

    SUCCESS = 0  # All tests were collected and passed successfully
    TESTS_FAILED = 1  # Tests were collected and run but some of the tests failed
    INTERRUPTED = 2  # Test execution was interrupted by the user
    INTERNAL_ERROR = 3  # Internal error happened while executing tests
    USAGE_ERROR = 4  # pytest command line usage error
    NO_TESTS_COLLECTED = 5  # No tests were collected


PytestErrorCodes = [
    PyTestExitCode.USAGE_ERROR,
    PyTestExitCode.INTERNAL_ERROR,
    PyTestExitCode.INTERRUPTED,
    PyTestExitCode.NO_TESTS_COLLECTED,
]


class PyTestReportSummary(BaseModel):
    """Summary statistics of a pytest run.

    Attributes:
        collected: Number of tests collected.
        passed: Number of tests passed.
        failed: Number of tests failed.
        xfailed: Number of tests expected to fail but passed.
        xpassed: Number of tests expected to pass but failed.
        error: Number of tests that errored.
        skipped: Number of tests skipped.
        total: Total number of tests.
    """

    collected: int = 0
    passed: int = 0
    failed: int = 0
    xfailed: int = 0
    xpassed: int = 0
    error: int = 0
    skipped: int = 0
    total: int = 0


class PyTestTestStage(BaseModel):
    """Details of a single pytest stage (setup, call, or teardown).

    Attributes:
        duration: Duration of stage in seconds.
        outcome: Outcome of the stage.
        crash: Crash entry.
        traceback: List of traceback entries.
        stdout: Standard output.
        stderr: Standard error.
        log: Log entries.
        longrepr: Representation of the error.
    """

    duration: float | None = None
    outcome: str | None = None
    crash: dict[str, Any] | None = None
    traceback: list[dict[str, Any]] | None = None
    stdout: str | None = None
    stderr: str | None = None
    log: list[dict[str, Any]] | None = None
    longrepr: str | None = None


class PyTestTest(BaseModel):
    """Details of a single pytest test.

    Attributes:
        nodeid: Node ID of the test.
        outcome: Outcome of the test.
        keywords: Keywords of the test.
        setup: Setup stage details.
        call: Call stage details.
        teardown: Teardown stage details.
        metadata: Metadata of the test.
    """

    nodeid: str
    outcome: str
    keywords: list[str] | None = None
    setup: PyTestTestStage | None = None
    call: PyTestTestStage | None = None
    teardown: PyTestTestStage | None = None
    metadata: dict[str, Any] | None = None


class PyTestReport(BaseModel):
    """Full report of a pytest run.

    Attributes:
        exitcode: Exit code of the full pytest run.
        duration: Duration of the test in seconds.
        root: Root directory of the test.
        summary: Summary of the test.
        tests: Details of tests in the report.
    """

    exitcode: PyTestExitCode
    duration: float | None = None
    root: str | None = None
    summary: PyTestReportSummary | None = None
    tests: list[PyTestTest] | None = Field(default=None, repr=False)

    def as_pandas_series(self) -> "Series":
        """Return the pytest report in a pandas representation."""
        import pandas as pd

        return pd.json_normalize(self.model_dump(exclude={"tests"}), sep="_").iloc[0]
