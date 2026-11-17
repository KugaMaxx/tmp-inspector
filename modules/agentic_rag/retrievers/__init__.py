"""Retriever builders for rag_eval_v2.

Each submodule registers its builder into ``REGISTRY`` on import, so importing
this package is enough to make every builder discoverable.
"""
from . import carag, fixed, graphrag, hierarchical, hyde, raptor, semantic  # noqa: F401
from .base import REGISTRY, load_or_build_index

__all__ = ["REGISTRY", "load_or_build_index"]
