import logging
from pathlib import Path
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from tmp_inspector.rag.modules.ca_rag import ClauseAwareRetriever, ClauseNodeParser, MineruDirectoryReader

from .base import load_or_build_index, register

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
    """Clause-aware hybrid retrieval with citation expansion, over MinerU documents.

    ``seed_k`` clauses come from the first round, ``top_k`` when unset. Cited
    clauses inherit ``hop_decay`` of their parent's score per hop and are
    ranked together with them. MinerU reads ``input_dir`` only when neither the
    index nor the clause nodes are cached, with the pages to skip from
    ``corpus_csv`` and its layout cached to ``mineru_cache_dir``.
    """
    nodes_path = index_dir / "clause_nodes.json"

    def build():
        if nodes_path.exists():
            # Parsing calls an LLM per clause, so its nodes are cached apart from the index
            logger.info(f"Loading clause nodes from {nodes_path}")
            nodes = list(SimpleDocumentStore.from_persist_path(str(nodes_path)).docs.values())
        else:
            documents = MineruDirectoryReader(
                input_dir=str(input_dir),
                corpus_csv=str(corpus_csv) if corpus_csv else None,
                cache_dir=str(mineru_cache_dir) if mineru_cache_dir else None,
                tier=tier,
            ).load_data()
            nodes = ClauseNodeParser(model_name=parse_model).get_nodes_from_documents(documents, show_progress=True)
            store = SimpleDocumentStore()
            store.add_documents(nodes)
            store.persist(persist_path=str(nodes_path))
        return VectorStoreIndex(nodes, show_progress=True)

    index = load_or_build_index(index_dir, build)
    # The persisted index docstore already holds the clause nodes
    nodes = list(index.storage_context.docstore.docs.values())
    return ClauseAwareRetriever(
        index, nodes,
        similarity_top_k=seed_k or top_k,
        max_hops=max_hops,
        hop_decay=hop_decay,
        max_cites_per_node=max_cites_per_node,
    )
