"""The inference gateway: the only path from a task container to a model.

Runs as its own container beside ``main`` and ``eval-sidecar``. It holds the one
real upstream credential and hands each consumer a separate scoped token, so the
optimizer cannot spend from the evaluation or finalization pools. Every proxied
request is checked three ways: the presented token against that scope's digest,
the requested model against the scope's allow-list, and the running budget
against a ledger written through to disk.

Provider-agnostic by design — it accepts either ``Authorization: Bearer`` or
``x-api-key``, and forwards any endpoint to the upstream proxy.
"""
