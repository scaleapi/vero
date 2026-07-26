"""Durable evaluation state.

``persistence`` holds evaluation records, manifests, and the partitioned case
store; ``budget`` holds the ledger that caps runs and cases per partition. Both
write through to disk so a restarted process resumes with its limits intact.
"""
