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
        default="/home/dszh/workspace/tmp-inspector/data/RAG/CoP",
        help="Directory containing the corpus files."
    )
    parser.add_argument(
        "--corpus_csv",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/configs/corpus.csv",
        help="CSV with columns: code,name,skip. Files absent from it are parsed in full."
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

    # Clause analysis settings
    parser.add_argument(
        "--parse_model",
        type=str,
        default=None,
        help="Optional Hugging Face text-generation model for topic and reference extraction."
    )

    # Embedding settings
    parser.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model used for the dense half of the retrieval."
    )

    # Query settings
    parser.add_argument(
        "--question",
        type=str,
        default="When may a building be permitted to have only one required staircase?", # Clause B6.1
        help="Retrieve clauses for this question and print them. No LLM is called."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of clauses kept after fusing the sparse and dense rankings."
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=1,
        help="Citation hops to follow from the retrieved clauses; 0 disables the expansion."
    )

    return parser.parse_args()


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
        model_name="Qwen/Qwen3-4B-Instruct-2507",
    ).get_nodes_from_documents(documents, show_progress=True)

    # Persist the index
    logger.info(f"Building and persisting the index to {args.index_dir}...")
    index = VectorStoreIndex(nodes, show_progress=True)
    # index.storage_context.persist(persist_dir=args.index_dir)

    # Retrieve for a question, printing the clauses without asking an LLM
    if args.question:
        retriever = ClauseAwareRetriever(
            index, nodes, similarity_top_k=args.top_k, expand=not args.no_expand
        )
        results = retriever.retrieve(args.question)

        # Print the retrieved clauses
        print(f"\nquestion: {args.question}\n")
        for hit in results:
            meta = hit.node.metadata
            print(f"[hop {meta['hop']} | {meta['relation']}]  score={hit.score or 0.0:.4f}")
            if meta["via"]:
                print(f"  via  : {meta['via']}")
            print(f"  topic: {meta['topic']}")
            print(f"  hier : {meta['hier']}")
            print(f"  local_cites   : {meta.get('local_cites', [])}")
            print(f"  external_cites: {meta.get('external_cites', [])}")
            print(f"  content: {meta['content'][:300]}...\n")


if __name__ == "__main__":
    main()
