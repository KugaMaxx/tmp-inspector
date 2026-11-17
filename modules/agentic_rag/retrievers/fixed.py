from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.retrievers import BaseRetriever

from .base import load_or_build_index, register


@register("fixed")
def build_fixed(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
) -> BaseRetriever:
    def build():
        documents = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data()
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)

    return load_or_build_index(index_dir, build).as_retriever(similarity_top_k=top_k)
