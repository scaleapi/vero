import asyncio


class FileEditException(Exception):
    """Base exception for file edit errors."""

    pass


class NoFilesChangedError(FileEditException):
    """Error raised when no files are changed."""

    pass


class InputTooLongError(FileEditException):
    """Error raised when the input is too long."""

    pass


class FileNotTrackedError(FileEditException):
    """Error raised when a file is not tracked by a repository."""

    pass


class StringNotFoundError(FileEditException):
    """Error raised when a string is not found in a file."""

    pass


class AccessDeniedError(PermissionError):
    """Error raised when access is denied."""

    pass


class CommitNotInBranchHistory(PermissionError):
    """Error raised when a commit is not in the allowed branch history."""

    pass


class StreamEventTimeout(asyncio.TimeoutError):
    pass
