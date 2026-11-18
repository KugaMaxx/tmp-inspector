from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.retrievers import BaseRetriever, TransformRetriever

from .base import register


@register("hyde")
def build_hyde(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
) -> BaseRetriever:
    """HyDE: hybrid retrieval with document expansion using LLMs"""

    # Initialize index directory
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load existing index from storage
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir)),
        )
    except FileNotFoundError:
        # Read documents from the input directory
        documents = SimpleDirectoryReader(
            input_dir=str(input_dir),
            required_exts=[".pdf"],
            recursive=True,
        ).load_data()
        
        # Split documents into chunks
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Create a vector store index
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
            show_progress=True,
        )
        index.storage_context.persist(str(index_dir))
        
    return TransformRetriever(
        index.as_retriever(similarity_top_k=top_k),
        HyDEQueryTransform(
            llm=Settings.llm,
            include_original=True,
        ),
    )
