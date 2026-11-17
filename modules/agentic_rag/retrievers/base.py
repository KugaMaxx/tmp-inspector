from typing import Callable, Dict

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
