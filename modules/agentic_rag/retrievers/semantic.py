from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import BaseRetriever

from .base import load_or_build_index, register


@register("semantic")
def build_semantic(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    threshold: int = 95,
) -> BaseRetriever:
    def build():
        documents = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data()
        splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model, breakpoint_percentile_threshold=threshold)
        return VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)

    return load_or_build_index(index_dir, build).as_retriever(similarity_top_k=top_k)
