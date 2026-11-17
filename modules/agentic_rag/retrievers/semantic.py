from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import BaseRetriever

from .base import register


@register("semantic")
def build_semantic(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    threshold: int = 95,
) -> BaseRetriever:
    """Semantic RAG: chunking based on semantic similarity with a percentile threshold for splitting"""

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
        splitter = SemanticSplitterNodeParser(
            embed_model=Settings.embed_model,
            breakpoint_percentile_threshold=threshold,
        )
        
        # Create a vector store index
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
            show_progress=True,
        )
        index.storage_context.persist(str(index_dir))

    return index.as_retriever(similarity_top_k=top_k)
