from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from ..utils import load_or_build_index


def build_graphrag(
    documents: Sequence[Document], index_dir: Path, top_k: int, llm: LLM, embed_model,
    chunk_size: int, max_paths: int,
) -> BaseRetriever:
    """llama_index PropertyGraphIndex with its default retriever.

    Entities are matched by LLM keyword synonyms and by embedding; each result
    is a source chunk with its matched triplets prepended.
    """
    index = load_or_build_index(
        index_dir,
        lambda: PropertyGraphIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            kg_extractors=[SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=max_paths)],
            transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)],
            show_progress=True,
        ),
        llm=llm,
        embed_model=embed_model,
    )
    return index.as_retriever(include_text=True, similarity_top_k=top_k)
