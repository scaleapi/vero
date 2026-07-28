"""Shared base model for VeRO's declarative contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects unknown fields.

    Authored configuration and on-disk records both inherit this, so a typo'd
    key fails loudly at load time instead of being dropped in silence and
    leaving a default in its place.
    """

    model_config = ConfigDict(extra="forbid")
