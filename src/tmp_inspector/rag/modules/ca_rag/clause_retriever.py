from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.vector_store import VectorStoreIndex
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


def _hybrid(
    index: VectorStoreIndex,
    nodes: List[BaseNode],
    top_k: int,
    code: Optional[str] = None,
) -> QueryFusionRetriever:
    """BM25 + dense fused with RRF, scoped to one document when ``code`` is given."""
    filters = (
        MetadataFilters(filters=[MetadataFilter(key="code", value=code)])
        if code else None
    )
    return QueryFusionRetriever(
        retrievers=[
            index.as_retriever(similarity_top_k=top_k, filters=filters),
            BM25Retriever.from_defaults(
                nodes=nodes, similarity_top_k=min(top_k, len(nodes))
            ),
        ],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=top_k,
        num_queries=1,  # no LLM query rewriting, fuse the query as given
        # num_queries=1 never calls the LLM, but the constructor still falls
        # back to Settings.llm, which resolves to OpenAI and demands a key.
        llm=MockLLM(),
        use_async=False,
    )


class ClauseAwareRetriever(BaseRetriever):
    """Retrieve clauses for a query, then follow their citations for ``max_hops`` hops.

    Args:
        index: vector index holding the clause nodes.
        nodes: the same clause nodes, used for BM25 and citation resolution.
        similarity_top_k: clauses kept by the first round, and per cited
            document by the second.
        max_hops: citation hops to follow; 0 disables the expansion.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: Sequence[BaseNode],
        similarity_top_k: int = 5,
        max_hops: int = 1,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._max_hops = max_hops
        nodes = list(nodes)
        self._fusion = _hybrid(index, nodes, similarity_top_k)

        by_doc: Dict[str, List[BaseNode]] = defaultdict(list)
        for node in nodes:
            by_doc[node.metadata.get("code", "")].append(node)
        by_doc.pop("", None)

        # One scoped retriever per document, for the second round
        self._doc_retrievers = {
            code: _hybrid(index, doc_nodes, similarity_top_k, code=code)
            for code, doc_nodes in by_doc.items()
        }
        self._doc_tokens = {code: _tokens(code) for code in by_doc}

        # (code, label) -> every clause filed under that heading, in reading order
        self._members: Dict[Tuple[str, str], List[BaseNode]] = defaultdict(list)
        for node in nodes:
            code = node.metadata.get("code", "")
            for label in node.metadata.get("labels", []):
                self._members[(code, label.upper())].append(node)

        super().__init__(callback_manager=callback_manager, verbose=verbose)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        seeds = [
            self._tag(hit.node, hit.score or 0.0, 0, None, "query")
            for hit in self._fusion.retrieve(query_bundle)
        ]
        return self._expand(query_bundle, seeds)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # BM25 is synchronous anyway
        return self._retrieve(query_bundle)

    def _expand(
        self, query_bundle: QueryBundle, seeds: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Breadth-first over the citations, one hop per frontier."""
        results = list(seeds)
        seen = {hit.node.node_id for hit in seeds}
        frontier = seeds

        for hop in range(1, self._max_hops + 1):
            reached: List[NodeWithScore] = []
            for parent in frontier:
                for hit, relation in self._follow(query_bundle, parent.node):
                    if hit.node.node_id in seen:
                        continue
                    seen.add(hit.node.node_id)
                    reached.append(self._tag(
                        hit.node, hit.score or 0.0, hop,
                        parent.node.metadata.get("hier"), relation,
                    ))
            if not reached:
                break
            results.extend(reached)
            frontier = reached
        return results

    def _follow(
        self, query_bundle: QueryBundle, source: BaseNode
    ) -> Iterator[Tuple[NodeWithScore, str]]:
        """Clauses reached through the citations of one clause."""
        code = source.metadata.get("code", "")

        # Local citations: every clause under the cited heading of this document
        for label in source.metadata.get("local_cites", []):
            for node in self._members.get((code, label.strip().upper()), []):
                # cited clauses are supporting context, not query matches
                yield NodeWithScore(node=node, score=0.0), "local"

        # External citations: second round scoped to the cited document
        query = QueryBundle(
            f"{query_bundle.query_str}\n{source.metadata.get('topic', '')}"
        )
        for cite in source.metadata.get("external_cites", []):
            target = self._resolve_doc(cite)
            if target is None or target == code:
                continue
            for hit in self._doc_retrievers[target].retrieve(query):
                yield hit, "external"

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
        node: BaseNode, score: float, hop: int, via: Optional[str], relation: str
    ) -> NodeWithScore:
        """Copy the node with its provenance, leaving the indexed node untouched."""
        keys = ["hop", "via", "relation"]
        node = node.model_copy()
        node.metadata = {**node.metadata, "hop": hop, "via": via, "relation": relation}
        node.excluded_embed_metadata_keys = [*node.excluded_embed_metadata_keys, *keys]
        node.excluded_llm_metadata_keys = [*node.excluded_llm_metadata_keys, *keys]
        return NodeWithScore(node=node, score=score)
