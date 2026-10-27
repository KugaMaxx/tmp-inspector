from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from llama_index.core import StorageContext, load_index_from_storage

logger = logging.getLogger(__name__)


def load_or_build_index(index_dir: Path, build: Callable, **load_kwargs):
    """Load an index from ``index_dir``, building and persisting it when absent."""
    if (index_dir / "docstore.json").exists():
        logger.info(f"Loading index from {index_dir}")
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir)), **load_kwargs
        )
    index = build()
    index.storage_context.persist(persist_dir=str(index_dir))
    logger.info(f"Index persisted to {index_dir}")
    return index
