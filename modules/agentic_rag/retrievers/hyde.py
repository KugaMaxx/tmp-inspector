from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.retrievers import BaseRetriever, TransformRetriever

from .base import load_or_build_index, register


@register("hyde")
def build_hyde(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> BaseRetriever:
    def build():
        documents = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data()
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)

    index = load_or_build_index(index_dir, build)
    return TransformRetriever(
        index.as_retriever(similarity_top_k=top_k),
        HyDEQueryTransform(llm=Settings.llm, include_original=True),
    )
