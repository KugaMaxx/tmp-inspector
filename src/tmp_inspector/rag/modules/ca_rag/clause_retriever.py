from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever


def _tokens(identifier: str) -> Tuple[str, ...]:
    """'BS EN ISO 1182:2010' -> ('BS', 'EN', 'ISO', '1182', '2010')."""
    text = "".join(ch if ch.isalnum() else " " for ch in identifier.upper())
    return tuple(text.split())


def _contains(outer: Tuple[str, ...], inner: Tuple[str, ...]) -> bool:
    """Whether ``inner`` is a contiguous run of ``outer``."""
    n = len(inner)
    return n > 0 and any(outer[i:i + n] == inner for i in range(len(outer) - n + 1))


def _rff_retriever(
    index: VectorStoreIndex,
    nodes: List[BaseNode],
    top_k: int,
    code: Optional[str] = None,
    node_ids: Optional[List[str]] = None,
) -> QueryFusionRetriever:
    """BM25 + dense fused with RRF over nodes."""
    filters = (
        MetadataFilters(filters=[MetadataFilter(key="code", value=code)])
        if code else None
    )
    return QueryFusionRetriever(
        retrievers=[
            VectorIndexRetriever(
                index, similarity_top_k=top_k, filters=filters,
                node_ids=node_ids or list(index.index_struct.nodes_dict.values()),
                callback_manager=index._callback_manager,
                object_map=index._object_map,
            ),
            BM25Retriever.from_defaults(
                nodes=nodes, similarity_top_k=min(top_k, len(nodes))
            ),
        ],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=top_k,
        num_queries=1,
        llm=MockLLM(),
        use_async=False,
    )


class ClauseAwareRetriever(BaseRetriever):
    """Retrieve clauses for a query, then follow their citations within multiple hops.

    Args:
        index: vector index holding the clause nodes.
        nodes: the same clause nodes, used for BM25 and citation resolution.
        similarity_top_k: clauses kept by the first round, and per cited
            document by the second.
        max_hops: citation hops to follow; 0 disables the expansion.
        hop_decay: share of its parent's score a cited clause inherits per hop.
        max_cites_per_node: cited clauses kept per clause and hop, best ``q`` first.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: Sequence[BaseNode],
        similarity_top_k: int = 5,
        max_hops: int = 1,
        hop_decay: float = 0.95,
        max_cites_per_node: int = 3,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._max_hops = max_hops
        self._hop_decay = hop_decay
        self._max_cites_per_node = max_cites_per_node
        self._index = index
        self._top_k = similarity_top_k
        nodes = list(nodes)
        self._rff_retriever = _rff_retriever(index, nodes, similarity_top_k)

        # Build the index of clauses by (code, heading)
        members: Dict[Tuple[str, str], List[BaseNode]] = defaultdict(list)
        for node in nodes:
            code = node.metadata.get("code", "")
            if not code:
                continue
            for head in ["", *(node.metadata.get("hier") or [])]:
                members[(code, head.strip().lower())].append(node)
        self._members = dict(members)
        
        self._doc_tokens = {code: _tokens(code) for code, head in self._members if not head}

        # Build the cached retrievers for multi-hop expansion
        self._scope_retrievers: Dict[Tuple[str, str], QueryFusionRetriever] = {}

        super().__init__(
            callback_manager=callback_manager, 
            verbose=verbose
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        seeds = [
            self._tag(hit.node, hit.score or 0.0, 0, None)
            for hit in self._rff_retriever.retrieve(query_bundle)
        ]
        return self._expand(query_bundle, seeds)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # BM25 is synchronous anyway
        return self._retrieve(query_bundle)

    def _expand(
        self, query_bundle: QueryBundle, seeds: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Breadth-first over the citations, one hop per frontier."""
        seed_ids = {hit.node.node_id for hit in seeds}
        best: Dict[str, Tuple[float, int, BaseNode, BaseNode]] = {}
        frontier = [(hit.node, hit.score or 0.0) for hit in seeds]

        # Multi-hop expansion
        for hop in range(1, self._max_hops + 1):
            improved: Dict[str, float] = {}

            # For each node in the frontier, find its children
            for parent, parent_score in frontier:
                children: Dict[str, Tuple[float, BaseNode]] = {}
                for node, q in self._follow(query_bundle, parent):
                    if node.node_id in seed_ids or node.node_id == parent.node_id:
                        continue
                    if node.node_id not in children or q > children[node.node_id][0]:
                        children[node.node_id] = (q, node)

                # Strongest children only, so one clause cannot flood the ranking
                ranked = sorted(children.values(), key=lambda c: c[0], reverse=True)
                for q, node in ranked[:self._max_cites_per_node]:
                    score = self._hop_decay * q * parent_score
                    if node.node_id not in best or score > best[node.node_id][0]:
                        best[node.node_id] = (score, hop, parent, node)
                        improved[node.node_id] = score

            # Only clauses whose score rose can raise the score of what they cite
            frontier = [(best[i][3], s) for i, s in improved.items()]
            if not frontier:
                break

        cited = [
            self._tag(node, score, hop, " > ".join(parent.metadata.get("hier") or []))
            for score, hop, parent, node in best.values()
        ]
        # Stable sort: on a tie, seeds before cited clauses, each in traversal order
        return sorted([*seeds, *cited], key=lambda hit: hit.score or 0.0, reverse=True)

    def _follow(
        self, query_bundle: QueryBundle, source: BaseNode
    ) -> Iterator[Tuple[BaseNode, float]]:
        """Clauses reached through the citations of one clause, with their ``q``."""
        code = source.metadata.get("code", "")

        query = QueryBundle(f"{query_bundle.query_str}\n{source.metadata.get('topic', '')}")
        for cite in source.metadata.get("cites", []):
            for doc, label in cite.items():
                target = code if doc.strip() == code else self._resolve_doc(doc)

                # The cited document is not in the corpus
                if not target:
                    continue

                # The label names a heading in the cited document, or is empty for the whole document
                key = (target, label.strip().lower())
                if key not in self._members:
                    key = (target, "")
                members = self._members[key]

                # If the label names a single clause (i.e., cite a specific clause)
                if len(members) == 1:
                    yield members[0], 1.0
                    continue

                # Find the most relevant clauses within a scope
                hits = self._scope_retriever(key).retrieve(query)
                top = max((hit.score or 0.0 for hit in hits), default=0.0)
                for hit in hits:
                    yield hit.node, (hit.score or 0.0) / top if top > 0 else 0.0

    def _scope_retriever(self, key: Tuple[str, str]) -> QueryFusionRetriever:
        """Hybrid retriever over the clauses under one heading, cached per heading."""
        if key not in self._scope_retrievers:
            members = self._members[key]
            self._scope_retrievers[key] = _rff_retriever(
                self._index, members, self._top_k,
                code=key[0], node_ids=[node.node_id for node in members],
            )
        return self._scope_retrievers[key]

    def _resolve_doc(self, cite: str) -> Optional[str]:
        """Corpus document named by an external citation, or None when absent."""
        tokens = _tokens(cite)
        hits = [
            (len(code_tokens), code)
            for code, code_tokens in self._doc_tokens.items()
            if _contains(tokens, code_tokens) or code_tokens[:len(tokens)] == tokens
        ]
        return max(hits)[1] if hits else None

    @staticmethod
    def _tag(
        node: BaseNode, score: float, hop: int, via: Optional[str]
    ) -> NodeWithScore:
        """Copy the node with its provenance, leaving the indexed node untouched."""
        keys = ["hop", "via"]
        node = node.model_copy()
        node.set_content(node.metadata.get("content") or node.get_content())
        node.metadata = {**node.metadata, "hop": hop, "via": via}
        node.excluded_embed_metadata_keys = [*node.excluded_embed_metadata_keys, *keys]
        # content is the text now, and topic is retrieval-only
        node.excluded_llm_metadata_keys = [
            *node.excluded_llm_metadata_keys, *keys, "content", "topic",
        ]
        return NodeWithScore(node=node, score=score)
