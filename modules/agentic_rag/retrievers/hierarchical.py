from pathlib import Path
from typing import Sequence

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever

from .base import register


@register("hierarchical")
def build_hierarchical(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_sizes: Sequence[int] = (2048, 512, 128),
) -> BaseRetriever:
    """Hirarchical RAG: multi-level chunking with auto-merging retrieval"""

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

        # Split documents into hierarchical chunks
        nodes = HierarchicalNodeParser.from_defaults(
            chunk_sizes=list(chunk_sizes),
        ).get_nodes_from_documents(documents,show_progress=True)

        # Parents must live in the docstore so leaf hits can be merged up into them
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(
            get_leaf_nodes(nodes),
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(str(index_dir))
        
    return AutoMergingRetriever(
        index.as_retriever(similarity_top_k=top_k),
        storage_context=index.storage_context,
    )
