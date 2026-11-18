import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrievers import REGISTRY
from tmp_inspector.rag.metrics import GenerationMetrics, RetrievalMetrics


ANSWER_PROMPT = PromptTemplate(
    "Answer the question using only the regulation excerpts below. "
    "Answer concisely in one to three sentences. If the excerpts do not contain "
    "the answer, say that the answer is not found.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Question: {query}\n"
    "Answer: "
)

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
        required=True,
        help="Read documents recursively in every subdirectory",
    )
    parser.add_argument(
        "--corpus_csv",
        type=str,
        required=True,
        help="CSV file that store the basic information about the documents.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cached embeddings and retrieval index will be loaded if provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval",
        help="Logs, answers and result tables. Cached answers are reused on rerun.",
    )

    # Systems
    parser.add_argument(
        "--systems",
        nargs="+",
        default=sorted(REGISTRY),
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
        default="cuda",
        help="Device to use for training (e.g., 'cuda' or 'cpu')"
    )

    # QA set
    parser.add_argument(
        "--qa_path",
        type=str,
        required=True,
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
        default="qwen3-8b",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed forwarded to vLLM for reproducibility.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max new tokens for the answering LLM.",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
        temperature=0.0,
        max_tokens=args.max_tokens,
        additional_kwargs={
            "seed": args.seed,
            "top_p": 1.0,
            "extra_body": {
                "top_k": 1,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
    )

    # Load Q&A questions
    logger.info(f"Loading Q&A questions from {args.qa_path}")
    items = load_qa_questions(args.qa_path)

    # Load RAG systems
    logger.info("Building RAG systems...")
    systems = build_systems(args)

    # Evaluate retrieval and (optionally) generation in a single pass
    logger.info(f"Evaluating retrieval...")
    metrics_ret = RetrievalMetrics()
    metrics_gen = GenerationMetrics()
    for name, retriever in systems.items():
        metrics_ret.reset()
        metrics_gen.reset()
        for item in tqdm(items, desc=f"Evaluating {name}"):
            hits = retriever.retrieve(item["question"])
            retrieval_hits = hits[:args.retrieval_k]

            print("question:\n", item["question"])
            print("ground truth:\n", "\n".join(item["gold"]))
            print("hits:\n", "\n".join([hit.node.get_content().replace("\\n", " ").replace("\r\n", " ").replace("\n", " ") for hit in retrieval_hits]))

            scores = metrics_ret.update([hit.node.get_content() for hit in retrieval_hits], item["gold"])
            print("retrieval scores:\n", scores)

            if metrics_gen is not None and (item.get("answer") or "").strip():
                context = "\n\n".join(hit.node.get_content() for hit in hits[:args.generation_k])
                prompt = ANSWER_PROMPT.format(query=item["question"], context=context)
                prediction = str(Settings.llm.complete(prompt)).strip()
                print("answer:\n", item["answer"])
                print("prediction:\n", prediction)
                scores = metrics_gen.update(prediction, item["answer"])
                print("generation scores:\n", scores)

            print('============================================')

        logger.info(f"\nRAG System: {name}\n Retrieval:\n {metrics_ret.summarize()}")
        if metrics_gen is not None:
            logger.info(f" Generation (w. {args.llm_model}):\n {metrics_gen.summarize()}")


if __name__ == "__main__":
    main()
