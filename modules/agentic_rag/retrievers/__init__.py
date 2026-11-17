"""Retriever builders for rag_eval_v2.

Each submodule registers its builder into ``REGISTRY`` on import, so importing
this package is enough to make every builder discoverable.
"""
from . import carag, empty, fixed, graphrag, hierarchical, hyde, raptor, semantic
from .base import REGISTRY

__all__ = ["REGISTRY"]
