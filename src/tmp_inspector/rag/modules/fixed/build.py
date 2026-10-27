from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document

from ..utils import load_or_build_index


def build_fixed(
    documents: Sequence[Document], index_dir: Path, top_k: int, chunk_size: int, chunk_overlap: int,
) -> BaseRetriever:
    """Dense retrieval over fixed-size token windows."""
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = load_or_build_index(
        index_dir, lambda: VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True),
    )
    return index.as_retriever(similarity_top_k=top_k)
