from pathlib import Path
from typing import Sequence

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever

from .base import load_or_build_index, register


@register("hierarchical")
def build_hierarchical(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_sizes: Sequence[int] = (2048, 512, 128),
) -> BaseRetriever:
    def build():
        documents = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data()
        nodes = HierarchicalNodeParser.from_defaults(chunk_sizes=list(chunk_sizes)).get_nodes_from_documents(documents, show_progress=True)
        # Parents must live in the docstore so leaf hits can be merged up into them
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        return VectorStoreIndex(get_leaf_nodes(nodes), storage_context=storage_context, show_progress=True)

    index = load_or_build_index(index_dir, build)
    return AutoMergingRetriever(index.as_retriever(similarity_top_k=top_k), storage_context=index.storage_context)
