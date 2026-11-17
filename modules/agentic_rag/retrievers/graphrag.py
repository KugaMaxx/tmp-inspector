import logging
from pathlib import Path
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    PropertyGraphIndex,
    SimpleLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode

from .base import load_or_build_index, register

logger = logging.getLogger(__name__)


class HybridGraphRAGRetriever(BaseRetriever):
    """Chunk-level dense retrieval as the primary signal, graph sub-retrievers as
    augmentation.

    PropertyGraphIndex.as_retriever concatenates sub-retriever results and ranks
    them by raw sub-retriever score. Entity cosine scores are not comparable to
    chunk-cosine scores, and the graph legs often push irrelevant chunks (whose
    entity names happen to match the query) above the right chunks. Here we
    run the two groups separately, min-max normalise within each group, give
    chunk hits a configurable weight, and merge by source chunk id.
    """

    def __init__(
        self,
        chunk_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        top_k: int,
        chunk_weight: float = 0.7,
    ):
        super().__init__()
        self._chunk_retriever = chunk_retriever
        self._graph_retriever = graph_retriever
        self._top_k = top_k
        self._chunk_weight = chunk_weight

    @staticmethod
    def _norm(hits):
        if not hits:
            return hits
        scores = [h.score or 0.0 for h in hits]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        for h in hits:
            h.score = 1.0 if span <= 1e-9 else ((h.score or 0.0) - lo) / span
        return hits

    def _retrieve(self, query_bundle):
        pool_k = max(self._top_k * 2, 20)
        chunk_hits = list(self._chunk_retriever.retrieve(query_bundle))[:pool_k]
        try:
            graph_hits = list(self._graph_retriever.retrieve(query_bundle))[:pool_k]
        except Exception as exc:
            logger.warning(f"graph sub-retrievers failed: {exc}")
            graph_hits = []

        self._norm(chunk_hits)
        self._norm(graph_hits)
        for h in chunk_hits:
            h.score = (h.score or 0.0) * self._chunk_weight
        for h in graph_hits:
            h.score = (h.score or 0.0) * (1.0 - self._chunk_weight)

        best = {}
        for h in chunk_hits + graph_hits:
            key = h.node.node_id
            if key not in best or (h.score or 0.0) > (best[key].score or 0.0):
                best[key] = h

        merged = sorted(best.values(), key=lambda h: h.score or 0.0, reverse=True)
        return merged[: self._top_k]


@register("graphrag")
def build_graphrag(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_size: int = 256,
    max_paths: int = 10,
    chunk_weight: float = 0.7,
) -> BaseRetriever:
    """Hybrid GraphRAG: chunk-level dense retrieval merged with the graph
    sub-retrievers of a PropertyGraphIndex."""
    index = load_or_build_index(
        index_dir,
        lambda: PropertyGraphIndex.from_documents(
            SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data(),
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=max_paths)],
            transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)],
            show_progress=True,
        ),
        llm=Settings.llm,
        embed_model=Settings.embed_model,
    )
    # SimplePropertyGraphStore does not support vector queries itself; the entity
    # embeddings live in the index's companion vector store, so wire it into
    # VectorContextRetriever explicitly, otherwise it silently returns zero hits.
    graph_store = index.property_graph_store
    graph_retriever = index.as_retriever(
        sub_retrievers=[
            LLMSynonymRetriever(graph_store, llm=llm, include_text=True),
            VectorContextRetriever(
                graph_store,
                vector_store=index.vector_store,
                embed_model=Settings.embed_model,
                include_text=True,
                similarity_top_k=top_k,
            ),
        ],
    )

    # Chunk-level dense retriever over the same source chunks the graph was
    # built on (persisted in the docstore). Cached beside the graph index.
    source_chunks = [n for n in index.storage_context.docstore.docs.values() if isinstance(n, TextNode)]
    chunk_index = load_or_build_index(
        index_dir / "_chunk_vector",
        lambda: VectorStoreIndex(source_chunks, embed_model=Settings.embed_model, show_progress=True),
        embed_model=Settings.embed_model,
    )
    chunk_retriever = chunk_index.as_retriever(similarity_top_k=top_k)

    return HybridGraphRAGRetriever(
        chunk_retriever=chunk_retriever,
        graph_retriever=graph_retriever,
        top_k=top_k,
        chunk_weight=chunk_weight,
    )
