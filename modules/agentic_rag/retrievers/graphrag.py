import logging
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    PropertyGraphIndex,
    SimpleLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever

from .base import register

logger = logging.getLogger(__name__)


class HybridGraphRAGRetriever(BaseRetriever):
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
        # Retrieve from both chunk-level and graph-level retrievers
        pool_k = max(self._top_k * 2, 20)
        chunk_hits = list(self._chunk_retriever.retrieve(query_bundle))[:pool_k]
        graph_hits = list(self._graph_retriever.retrieve(query_bundle))[:pool_k]

        # Score normalization and weighted sum
        self._norm(chunk_hits)
        self._norm(graph_hits)
        for h in chunk_hits:
            h.score = (h.score or 0.0) * self._chunk_weight
        for h in graph_hits:
            h.score = (h.score or 0.0) * (1.0 - self._chunk_weight)

        # Merge hits by node_id, keeping the best score for each node
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
    """Hybrid Graph RAG: combines a graph-based retriever with a chunk-based retriever, allowing for both structured and unstructured retrieval."""
    
    # Initialize index directory
    graph_index_dir = index_dir / "graph"
    chunk_index_dir = index_dir / "chunk"
    graph_index_dir.mkdir(parents=True, exist_ok=True)
    chunk_index_dir.mkdir(parents=True, exist_ok=True)

    # Graph-level index and retriever
    try:
        # Load existing index from storage
        graph_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(graph_index_dir)),
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )
    except FileNotFoundError:
        # Read documents from the input directory
        documents = SimpleDirectoryReader(
            input_dir=str(input_dir),
            required_exts=[".pdf"],
            recursive=True,
        ).load_data()
        
        # Create a property graph index with LLM-based path extraction
        graph_index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            kg_extractors=[
                SimpleLLMPathExtractor(
                    llm=Settings.llm,
                    max_paths_per_chunk=max_paths,
                ),
            ],
            transformations=[
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                ),
            ],
            show_progress=True,
        )
        graph_index.storage_context.persist(str(graph_index_dir))

    graph_store = graph_index.property_graph_store
    graph_retriever = graph_index.as_retriever(
        sub_retrievers=[
            LLMSynonymRetriever(
                graph_store,
                llm=Settings.llm,
                include_text=True,
            ),
            VectorContextRetriever(
                graph_store,
                vector_store=graph_index.vector_store,
                embed_model=Settings.embed_model,
                include_text=True,
                similarity_top_k=top_k,
            ),
        ],
    )

    # Chunk-level index and retriever
    try:
        chunk_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(chunk_index_dir)),
            embed_model=Settings.embed_model,
        )
    except FileNotFoundError:
        # Read documents from the input directory
        documents = SimpleDirectoryReader(
            input_dir=str(input_dir),
            required_exts=[".pdf"],
            recursive=True,
        ).load_data()
        
        # Split documents into chunks
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
        )

        # Create a vector store index
        chunk_index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
            embed_model=Settings.embed_model,
            show_progress=True,
        )
        chunk_index.storage_context.persist(str(chunk_index_dir))

    chunk_retriever = chunk_index.as_retriever(similarity_top_k=top_k)

    return HybridGraphRAGRetriever(
        chunk_retriever=chunk_retriever,
        graph_retriever=graph_retriever,
        top_k=top_k,
        chunk_weight=chunk_weight,
    )
