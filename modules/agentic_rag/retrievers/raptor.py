import logging
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode

from .base import load_or_build_index, register

logger = logging.getLogger(__name__)


RAPTOR_PROMPT = PromptTemplate(
    "Summarize the regulation excerpts below in one paragraph, keeping every "
    "requirement, condition and figure they state.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Summary: "
)


def _raptor_clusters(embeddings: np.ndarray, dim: int = 10, threshold: float = 0.1, seed: int = 224) -> List[np.ndarray]:
    import umap
    from sklearn.mixture import GaussianMixture

    def mixture(x: np.ndarray, n_neighbors: int) -> List[np.ndarray]:
        if len(x) <= dim + 1:
            return [np.arange(len(x))]
        x = umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric="cosine", random_state=seed).fit_transform(x)
        fits = [GaussianMixture(k, random_state=seed).fit(x) for k in range(1, min(50, len(x)))]
        best = min(fits, key=lambda gm: gm.bic(x))
        probs = best.predict_proba(x)
        return [np.where(probs[:, c] > threshold)[0] for c in range(best.n_components)]

    clusters = []
    for outer in mixture(embeddings, max(2, int((len(embeddings) - 1) ** 0.5))):
        clusters.extend(outer[inner] for inner in mixture(embeddings[outer], 10))
    return [cluster for cluster in clusters if len(cluster)]


@register("raptor")
def build_raptor(
    input_dir: Path,
    index_dir: Path,
    top_k: int,
    chunk_size: int = 256,
    tree_depth: int = 3,
) -> BaseRetriever:
    def build():
        documents = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".pdf"], recursive=True).load_data()
        level_nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0).get_nodes_from_documents(documents)
        summarizer = get_response_synthesizer(
            llm=Settings.llm, response_mode="tree_summarize", summary_template=RAPTOR_PROMPT,
        )

        tree = list(level_nodes)
        for level in range(1, tree_depth + 1):
            if len(level_nodes) <= 1:
                break
            embeddings = Settings.embed_model.get_text_embedding_batch(
                [node.get_content() for node in level_nodes], show_progress=True,
            )
            for node, embedding in zip(level_nodes, embeddings):
                node.embedding = embedding

            summaries = []
            for cluster in tqdm(_raptor_clusters(np.array(embeddings)), desc=f"raptor:level {level}"):
                text = summarizer.get_response("Summarize the excerpts.", [level_nodes[i].get_content() for i in cluster])
                summaries.append(TextNode(
                    text=str(text).strip(), metadata={"level": level},
                    excluded_embed_metadata_keys=["level"], excluded_llm_metadata_keys=["level"],
                ))
            logger.info(f"RAPTOR level {level}: {len(level_nodes)} nodes into {len(summaries)} summaries")
            tree.extend(summaries)
            level_nodes = summaries
        return VectorStoreIndex(tree, show_progress=True)

    return load_or_build_index(index_dir, build).as_retriever(similarity_top_k=top_k)
