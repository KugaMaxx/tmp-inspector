import logging
from pathlib import Path
from typing import Callable, Dict

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

logger = logging.getLogger(__name__)

# name -> builder(input_dir, index_dir, top_k, **kwargs) -> BaseRetriever
REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    """Register a builder under ``name``; collisions are a programming error."""
    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"retriever '{name}' already registered")
        REGISTRY[name] = fn
        return fn
    return decorator


def load_or_build_index(index_dir: Path, build_fn, **load_kwargs) -> VectorStoreIndex:
    """Load an index from disk or build it if not present."""
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(index_dir)), **load_kwargs)
    except FileNotFoundError:
        logger.info(f" index not found in {index_dir}, building...")
        index = build_fn()
        index.storage_context.persist(str(index_dir))
        return index
