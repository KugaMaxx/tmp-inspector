from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document

from ..utils import load_or_build_index


def build_semantic(
    documents: Sequence[Document], index_dir: Path, top_k: int, embed_model, threshold: int,
) -> BaseRetriever:
    """Dense retrieval over chunks cut where adjacent sentences drift apart in embedding space."""
    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model, buffer_size=1, breakpoint_percentile_threshold=threshold,
    )
    index = load_or_build_index(
        index_dir, lambda: VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True),
    )
    return index.as_retriever(similarity_top_k=top_k)
