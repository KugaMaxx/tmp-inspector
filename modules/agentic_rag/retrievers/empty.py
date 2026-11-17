from pathlib import Path

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .base import register


class EmptyRetriever(BaseRetriever):
    def __init__(self, top_k: int):
        super().__init__()
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle):
        return [NodeWithScore(node=TextNode(text=""), score=0.0) for _ in range(self._top_k)]


@register("empty")
def build_empty(input_dir: Path, index_dir: Path, top_k: int) -> BaseRetriever:
    return EmptyRetriever(top_k=top_k)
