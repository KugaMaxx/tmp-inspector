from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.docstore import SimpleDocumentStore

from ..utils import load_or_build_index
from .clause_parser import ClauseNodeParser
from .clause_retriever import ClauseAwareRetriever

logger = logging.getLogger(__name__)


def _clause_nodes(documents: Sequence[Document], nodes_path: Path, parse_model: str) -> List[BaseNode]:
    """Clause nodes, cached to ``nodes_path`` since parsing calls an LLM per clause."""
    if nodes_path.exists():
        logger.info(f"Loading clause nodes from {nodes_path}")
        return list(SimpleDocumentStore.from_persist_path(str(nodes_path)).docs.values())
    nodes = ClauseNodeParser(model_name=parse_model).get_nodes_from_documents(documents, show_progress=True)
    store = SimpleDocumentStore()
    store.add_documents(nodes)
    store.persist(persist_path=str(nodes_path))
    return nodes


def build_carag(
    documents: Sequence[Document], index_dir: Path, top_k: int, parse_model: str,
    max_hops: int, seed_k: Optional[int],
) -> BaseRetriever:
    """Clause-aware hybrid retrieval with citation expansion, over MinerU documents.

    ``seed_k`` clauses come from the first round and citation hops fill the
    ranking after them; left unset, the first round alone fills ``top_k``.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    nodes = _clause_nodes(documents, index_dir / "clause_nodes.json", parse_model)
    index = load_or_build_index(index_dir, lambda: VectorStoreIndex(nodes, show_progress=True))
    return ClauseAwareRetriever(index, nodes, similarity_top_k=seed_k or top_k, max_hops=max_hops)
