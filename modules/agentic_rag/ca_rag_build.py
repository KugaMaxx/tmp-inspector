"""Build the clause-aware RAG index of a regulatory corpus.

    python -m modules.agentic_rag.carag_build \
        --corpus_dir data/RAG/test \
        --question "What is the minimum width of an exit door?"

Documents are laid out by MinerU, chunked into one node per normative clause,
embedded and persisted. Passing ``--cache_dir`` caches the layout analysis per
document, so only the first run pays for it. Passing ``--question`` retrieves
clauses and prints them as they are stored, without asking an LLM.
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Allow this file to be launched directly as well as with ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.agentic_rag.ca_rag import (
    ClauseNodeParser,
    MineruDirectoryReader,
    ClauseAwareRetriever,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a clause-aware RAG index from a corpus of regulations.")
    # Corpus settings
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/data/RAG/test",
        help="Directory containing the corpus files."
    )
    parser.add_argument(
        "--corpus_csv",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/configs/corpus.csv",
        help="CSV with columns: file_name,doc_code,skip_pages. Files absent from it are parsed in full."
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/outputs/carag_index",
        help="Directory to persist the built index. An existing index is loaded instead of rebuilt."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/outputs/carag_cache",
        help="Directory caching the MinerU layout analysis, one JSON per document. "
             "Left unset, documents are laid out on every run and nothing is cached."
    )
    parser.add_argument(
        "--force_reparse",
        action="store_true",
        help="Re-run MinerU even when a cached layout analysis exists."
    )

    # MinerU settings
    parser.add_argument(
        "--tier",
        type=str,
        default="standard",
        choices=["flash", "basic", "standard", "advanced"],
        help="MinerU parse tier; higher tiers read harder layouts at a lower speed."
    )

    # Chunking settings
    parser.add_argument(
        "--min_clause_chars",
        type=int,
        default=40,
        help="Clauses shorter than this are merged into the preceding one."
    )
    parser.add_argument(
        "--max_clause_chars",
        type=int,
        default=4000,
        help="Clauses longer than this are cut into successive parts."
    )

    # Embedding settings
    parser.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model used for the dense half of the retrieval."
    )
    # parser.add_argument(
    #     "--api_base",
    #     type=str,
    #     default=os.environ.get("OPENAI_API_BASE", "https://aihubmix.com/v1"),
    #     help="Base URL of an OpenAI-compatible embedding endpoint."
    # )
    # parser.add_argument(
    #     "--api_key",
    #     type=str,
    #     default=os.environ.get("OPENAI_API_KEY", ""),
    #     help="API key of the embedding endpoint. Prefer the OPENAI_API_KEY environment variable."
    # )

    # Query settings
    parser.add_argument(
        "--question",
        type=str,
        default="For sprinkler protected building, which should be provided?",
        help="Retrieve clauses for this question and print them. No LLM is called."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of clauses kept after fusing the sparse and dense rankings."
    )
    parser.add_argument(
        "--no_expand",
        action="store_true",
        help="Do not pull in the clauses cited by the retrieved ones."
    )

    return parser.parse_args()


def print_retrieved(question: str, results) -> None:
    """Print the clauses as they are stored, so the chunking can be judged."""
    print(f"\nquestion: {question}\n")
    if not results:
        print("no clauses retrieved")
        return

    for hit in results:
        kind = "matched" if hit.score else "cited by a match"
        score = f"{hit.score:.4f}" if hit.score else "-"
        print(f"[{kind}] {hit.node.metadata['ID']}  score={score}")
        print(f"  Topic: {hit.node.metadata['Topic']}")
        print(f"  Hier : {hit.node.metadata['Hier']}")
        print(f"  Ref  : {hit.node.metadata['Ref']}")
        print(f"  {hit.node.get_content(metadata_mode='none')[:300]}...\n")


def main():
    # Parse arguments
    args = parse_args()

    # Set logging
    logging_dir = Path(args.index_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(logging_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ],
    )
    logger.info(f"Starting script: {Path(__file__).name}")
    logger.info(f"Build Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items() if arg != 'api_key'])} \n")

    # Set the embedding model; the LLM is left unset because nothing here calls one
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3", device="cuda", trust_remote_code=True,
    )

    # Read the documents from the corpus
    logger.info(f"Reading documents from {args.corpus_dir}...")
    documents = MineruDirectoryReader(
        input_dir=args.corpus_dir,
        corpus_csv=args.corpus_csv,
        cache_dir=args.cache_dir,
        tier=args.tier,
        force_reparse=args.force_reparse,
    ).load_data()

    # Chunk the documents into clauses
    logger.info(f"Parsing clauses from documents...")
    nodes = ClauseNodeParser(
        min_clause_chars=args.min_clause_chars,
        max_clause_chars=args.max_clause_chars,
    ).get_nodes_from_documents(documents, show_progress=True)

    # Persist the index
    logger.info(f"Building and persisting the index to {args.index_dir}...")
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=args.index_dir)

    # Retrieve for a question, printing the clauses without asking an LLM
    if args.question:
        retriever = ClauseAwareRetriever(
            index, nodes, similarity_top_k=args.top_k, expand=not args.no_expand
        )
        print_retrieved(args.question, retriever.retrieve(args.question))


if __name__ == "__main__":
    main()
