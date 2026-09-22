"""Clause-aware retrieval: hybrid BM25 + dense search fused with RRF, then
expanded along the cross-references declared in each clause."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever


class ClauseAwareRetriever(BaseRetriever):
    """Retrieve clauses for a query and add the clauses they cite.

    Args:
        index: vector index holding the clause nodes.
        nodes: the same clause nodes, used for the BM25 index and for
            resolving references by their ``ID``.
        similarity_top_k: number of clauses kept after fusion (``K``).
        expand: whether to follow the ``Ref`` of every retrieved clause.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: Sequence[BaseNode],
        similarity_top_k: int = 5,
        expand: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._top_k = similarity_top_k
        self._expand = expand
        self._nodes_by_id: Dict[str, BaseNode] = {
            node.metadata["ID"]: node for node in nodes if node.metadata.get("ID")
        }
        self._fusion = QueryFusionRetriever(
            retrievers=[
                index.as_retriever(similarity_top_k=similarity_top_k),
                BM25Retriever.from_defaults(
                    nodes=list(nodes), similarity_top_k=similarity_top_k
                ),
            ],
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=similarity_top_k,
            num_queries=1,  # no LLM query rewriting, fuse the query as given
            # num_queries=1 never calls the LLM, but the constructor still falls
            # back to Settings.llm, which resolves to OpenAI and demands a key.
            llm=MockLLM(),
            use_async=False,
            verbose=verbose,
        )
        super().__init__(callback_manager=callback_manager, verbose=verbose)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        initial = self._fusion.retrieve(query_bundle)
        return initial + self._cited(initial) if self._expand else initial

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        initial = await self._fusion.aretrieve(query_bundle)
        return initial + self._cited(initial) if self._expand else initial

    def _cited(self, retrieved: List[NodeWithScore]) -> List[NodeWithScore]:
        """One hop over the citation edges of the retrieved clauses."""
        seen = {node.node.metadata.get("ID") for node in retrieved}
        expanded: List[NodeWithScore] = []
        for node in retrieved:
            for ref_id in node.node.metadata.get("Ref", []):
                cited = self._nodes_by_id.get(ref_id)
                if cited is None or ref_id in seen:
                    continue
                seen.add(ref_id)
                # cited clauses are supporting context, not query matches
                expanded.append(NodeWithScore(node=cited, score=0.0))
        return expanded
