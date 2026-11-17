import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrievers import REGISTRY
from tmp_inspector.rag.metrics import GenerationMetrics, RetrievalMetrics

logger = logging.getLogger(__name__)


def load_qa_questions(qa_path):
    with open(qa_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for i, item in enumerate(items):
        item["id"] = f"q{i:04d}"
        item["gold"] = [ev["text"] for ev in item["evidence"] if (ev.get("text") or "").strip()]
    
    return items


def build_systems(args):
    top_k = max(args.retrieval_k, args.generation_k)
    corpus_dir, cache_dir = Path(args.corpus_dir), Path(args.cache_dir)

    systems = {}
    for i, name in enumerate(args.systems):
        logger.info(f" currently working on {name} [{i + 1}/{len(args.systems)}]")
        systems[name] = REGISTRY[name](input_dir=corpus_dir, index_dir=cache_dir / name, top_k=top_k)

    return systems


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval-augmented generation benchmark.")
    # Corpus settings
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/data/RAG",
        help="Read documents recursively in every subdirectory",
    )
    parser.add_argument(
        "--corpus_csv",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/configs/corpus.csv",
        help="CSV file that store the basic information about the documents.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/cache",
        help="Cached embeddings and retrieval index will be loaded if provided.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/eval",
        help="Logs, answers and result tables. Cached answers are reused on rerun.",
    )

    # Systems
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["fixed"],
        choices=sorted(REGISTRY),
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model shared by every system.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to use for training (e.g., 'cuda' or 'cpu')"
    )

    # QA set
    parser.add_argument(
        "--qa_path",
        type=str,
        default="/home/23132798r/workspace/tmp-inspector/data/RAG/test_questions.json",
        help="JSON array of {question, answer, evidence: [{source, text}], comments}. "
        "Synthesised from the plain text when the file is absent.",
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=10,
        help="Depth of the ranking that is scored.",
    )
    parser.add_argument(
        "--generation_k",
        type=int,
        default=5,
        help="Passages handed to the answering LLM.",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.5,
        help="Share of 3-grams a passage must share with a gold passage to count as relevant.",
    )

    # LLM settings
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen3-max",
        help="LLM model used for retrieval and generation.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:7300/v1",
        help="API URL for the LLM service.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="local",
        help="API key for the LLM service.",
    )
    parser.add_argument(
        "--skip_generation",
        default=True,
        action="store_true",
        help="Score retrieval only.",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(work_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ],
    )
    logger.info(f"Starting script: {Path(__file__).name}")
    logger.info(f"Eval Arguments: \n {'\n '.join([f'{arg}: {value}' for arg, value in vars(args).items()])} \n")

    # One embedding model for every system; LLMs are passed explicitly, never through Settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embed_model, 
        device=args.device, 
        trust_remote_code=True
    )
    Settings.llm = OpenAILike(
        model=args.llm_model, 
        api_base=args.api_base, 
        api_key=args.api_key, 
        is_chat_model=True,
        additional_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
    )

    # Load Q&A questions
    logger.info(f"Loading Q&A questions from {args.qa_path}")
    items = load_qa_questions(args.qa_path)

    # Load RAG systems
    logger.info("Building RAG systems...")
    systems = build_systems(args)

    # Evaluate retrieval
    logger.info("Evaluating retrieval...")
    metrics = RetrievalMetrics()
    for name, retriever in systems.items():
        metrics.reset()
        for item in tqdm(items, desc=f"Evaluating {name}"):
            hits = retriever.retrieve(item["question"])
            hits = hits[:args.retrieval_k]

            print("question:\n", item["question"])
            print("ground truth:\n", "\n".join(item["gold"]))
            print("hits:\n", "\n".join([hit.node.get_content().replace("\\n", " ").replace("\r\n", " ").replace("\n", " ") for hit in hits]))

            scores = metrics.update([hit.node.get_content() for hit in hits], item["gold"])
            print("scores:\n", scores)
            print('============================================')

        logger.info(f"\nRAG System: {name}\n {metrics.summarize()}")


if __name__ == "__main__":
    main()
