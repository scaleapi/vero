"""Pluggable evaluation execution.

``base`` defines the ``EvaluationBackend`` protocol and the registry that
resolves one; ``command`` and ``python_task`` are two implementations. A third,
``HarborBackend``, lives in ``vero.harbor.backend`` — it is a peer of these, kept
in that package because it depends on Harbor machinery that ``vero.evaluation``
must not.
"""
