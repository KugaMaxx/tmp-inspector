import logging
from pathlib import Path
from typing import Optional

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from tmp_inspector.rag.modules.ca_rag import (
    ClauseAwareRetriever,
    ClauseNodeParser,
    MineruDirectoryReader,
)

from .base import register

logger = logging.getLogger(__name__)


@register("carag")
def build_carag(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    corpus_csv: Optional[Path] = None,
    mineru_cache_dir: Optional[Path] = None,
    parse_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    seed_k: Optional[int] = None,
    max_hops: int = 1,
    hop_decay: float = 0.95,
    max_cites_per_node: int = 3,
    tier: str = "advanced",
) -> BaseRetriever:
    """Clause-aware hybrid retrieval with citation expansion"""

    # Initialize index directory
    index_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = index_dir / "clause_nodes.json"

    try:
        # Load existing index from storage
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir)),
        )
    except FileNotFoundError:
        if nodes_path.exists():
            # Parsing calls an LLM per clause, so its nodes are cached apart from the index
            logger.info(f"Loading clause nodes from {nodes_path}")
            nodes = list(SimpleDocumentStore.from_persist_path(str(nodes_path)).docs.values())
        else:
            # Read documents from the input directory
            documents = MineruDirectoryReader(
                input_dir=str(input_dir),
                corpus_csv=str(corpus_csv) if corpus_csv else None,
                cache_dir=str(mineru_cache_dir) if mineru_cache_dir else None,
                tier=tier,
            ).load_data()

            # Parse the documents into clause nodes
            nodes = ClauseNodeParser(
                model_name=parse_model,
            ).get_nodes_from_documents(documents, show_progress=True)
            
            # Create a document store for the clause nodes and persist it
            store = SimpleDocumentStore()
            store.add_documents(nodes)
            store.persist(persist_path=str(nodes_path))

        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(str(index_dir))

    nodes = list(index.storage_context.docstore.docs.values())

    return ClauseAwareRetriever(
        index,
        nodes,
        similarity_top_k=seed_k or top_k,
        max_hops=max_hops,
        hop_decay=hop_decay,
        max_cites_per_node=max_cites_per_node,
    )
