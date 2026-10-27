from __future__ import annotations

import os
import re
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field
from llama_index.core import Settings, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tmp_inspector.rag.modules import build_carag, build_fixed, build_graphrag, build_semantic
from tmp_inspector.rag.modules.ca_rag import MineruDirectoryReader
from tmp_inspector.rag.metrics import GenerationMetrics, RetrievalMetrics

logger = logging.getLogger(__name__)

SYSTEMS = ["carag"]


def make_llm(model: str, api_base: Optional[str], api_key: Optional[str], temperature: float = 0.0) -> LLM:
    """Any OpenAI-compatible chat model; the key falls back to ``OPENAI_API_KEY``."""
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        is_chat_model=True,
        context_window=32768,
        max_tokens=1024,
        timeout=120,
        max_retries=3,
    )


# --- QA set -------------------------------------------------------------------

class QAPair(BaseModel):
    question: str = Field(description="A question an inspector would ask, answerable from the excerpt alone.")
    answer: str = Field(description="A concise answer of one to three sentences, stated in the excerpt's own terms.")


QA_PROMPT = PromptTemplate(
    "You write evaluation questions for a retrieval system over fire safety regulations.\n"
    "Given the excerpt below, write ONE question whose answer is stated in the excerpt, "
    "and the reference answer. The question must make sense without seeing the excerpt: "
    "never say 'this excerpt' or 'the above', and do not quote clause numbers.\n\n"
    "Excerpt ({code}):\n{content}\n"
)


def generate_qa(
    documents: Sequence[Document], llm: LLM, num_questions: int,
    passage_size: int = 384, min_chars: int = 200, seed: int = 0,
) -> List[Dict]:
    """Synthesise one question per sampled passage, keeping the passage as gold evidence.

    Passages are cut by sentence, not by any evaluated system, so the gold
    boundaries favour none of them.
    """
    splitter = SentenceSplitter(chunk_size=passage_size, chunk_overlap=0)
    pool = [n for n in splitter.get_nodes_from_documents(documents) if len(n.get_content()) >= min_chars]
    random.Random(seed).shuffle(pool)

    items = []
    for node in pool:
        if len(items) >= num_questions:
            break
        code = node.metadata.get("code") or node.metadata.get("file_name", "")
        content = node.get_content()
        try:
            pair = llm.structured_predict(QAPair, QA_PROMPT, code=code, content=content)
        except Exception as exc:  # malformed JSON from the model, skip the passage
            logger.warning(f"Skipping a passage of {code}: {exc}")
            continue
        items.append({
            "question": pair.question,
            "answer": pair.answer,
            "evidence": [{"source": code, "text": content}],
            "comments": "synthetic",
        })
        logger.info(f"[{len(items)}/{num_questions}] {code}: {pair.question}")
    return items


def save_json(items: Sequence[Dict], path: Path) -> None:
    """A JSON array, indented so the file stays readable by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(items), f, ensure_ascii=False, indent=4)


def load_json(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path) -> List[Dict]:
    """The QA set: a JSON array of {question, answer, evidence: [{source, text}], comments}.

    Each item gets an ``id`` from its position and a ``gold`` list of its
    evidence texts. Evidence without text cites a document outside the corpus,
    which no system can retrieve, so it is left out of the gold.
    """
    items = load_json(path)
    for i, item in enumerate(items):
        item["id"] = f"q{i:04d}"
        item["gold"] = [ev["text"] for ev in item["evidence"] if (ev.get("text") or "").strip()]
    return items


# --- Answering ----------------------------------------------------------------

ANSWER_PROMPT = PromptTemplate(
    "Answer the question using only the regulation excerpts below. "
    "Answer concisely in one to three sentences. If the excerpts do not contain "
    "the answer, say that the answer is not found.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def answer(llm: LLM, question: str, nodes: Sequence[NodeWithScore]) -> str:
    """One answer from the retrieved nodes, with the same prompt for every system."""
    synthesizer = get_response_synthesizer(llm=llm, text_qa_template=ANSWER_PROMPT, response_mode="compact")
    return str(synthesizer.synthesize(question, nodes=list(nodes))).strip()


# --- Evaluation ---------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Compare clause-aware RAG against chunking and graph baselines.")
    # Corpus settings
    parser.add_argument("--corpus_dir", type=str, default="/home/dszh/workspace/tmp-inspector/data/RAG",
                        help="PDFs read recursively, every subdirectory included.")
    parser.add_argument("--corpus_csv", type=str, default="/home/dszh/workspace/tmp-inspector/configs/corpus.csv")
    parser.add_argument("--cache_dir", type=str, default="/home/dszh/workspace/tmp-inspector/outputs/carag_cache",
                        help="MinerU layout cache shared with ca_rag_build.py, plus one index folder per system "
                             "(fixed/, semantic/, graphrag/, carag/). An existing folder is loaded instead of rebuilt.")
    parser.add_argument("--work_dir", type=str, default="/home/dszh/workspace/tmp-inspector/outputs/rag_eval",
                        help="Logs, answers and result tables. Cached answers are reused on rerun.")

    # Systems
    parser.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3",
                        help="Embedding model shared by every system.")
    parser.add_argument("--chunk_size", type=int, default=256, help="Tokens per chunk for fixed and graphrag.")
    parser.add_argument("--chunk_overlap", type=int, default=32, help="Token overlap for fixed chunks.")
    parser.add_argument("--semantic_threshold", type=int, default=95,
                        help="Breakpoint percentile of the semantic splitter.")
    parser.add_argument("--graph_model", type=str, default="qwen3-max",
                        help="LLM extracting triplets and query keywords for graphrag.")
    parser.add_argument("--graph_max_paths", type=int, default=10, help="Triplets extracted per chunk.")
    parser.add_argument("--parse_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Local model the clause parser runs, as in ca_rag_build.py.")
    parser.add_argument("--max_hops", type=int, default=1, help="Citation hops of carag.")
    parser.add_argument("--carag_seed_k", type=int, default=None,
                        help="First-round clauses of carag, with hops filling the rest of the ranking. "
                             "Unset, the first round fills the whole top-k, as the baselines do.")

    # QA set
    parser.add_argument("--qa_path", type=str, default="/home/dszh/workspace/tmp-inspector/data/RAG/test_questions.json",
                        help="JSON array of {question, answer, evidence: [{source, text}], comments}. "
                             "Synthesised from the plain text when the file is absent.")
    parser.add_argument("--qa_model", type=str, default="qwen3-max", help="LLM writing the synthetic QA set.")
    parser.add_argument("--num_questions", type=int, default=100)

    # Evaluation
    parser.add_argument("--llm_models", nargs="+", default=["qwen3-max", "gpt-4o-mini", "deepseek-v3"],
                        help="Answering LLMs compared in the generation table.")
    parser.add_argument("--api_base", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1"))
    parser.add_argument("--retrieval_k", type=int, default=10, help="Depth of the ranking that is scored.")
    parser.add_argument("--generation_k", type=int, default=5, help="Passages handed to the answering LLM.")
    parser.add_argument("--match_threshold", type=float, default=0.5,
                        help="Share of 3-grams a passage must share with a gold passage to count as relevant.")
    parser.add_argument("--skip_generation", default=True, action="store_true", help="Score retrieval only.")

    return parser.parse_args()


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def read_plain(args):
    """The corpus as the baselines see it: llama_index's default PDF reader, one document per page."""
    return SimpleDirectoryReader(input_dir=args.corpus_dir, required_exts=[".pdf"], recursive=True).load_data()


def read_mineru(args):
    """The corpus as CA-RAG sees it: MinerU layout, with the pages to skip from corpus_csv."""
    return MineruDirectoryReader(
        input_dir=args.corpus_dir, corpus_csv=args.corpus_csv, cache_dir=args.cache_dir,
    ).load_data()


def build_systems(args, plain, depth: int):
    """Only the requested systems are built; each index is persisted to cache_dir/<system>."""
    cache_dir = Path(args.cache_dir)
    builders = {
        "fixed": lambda: build_fixed(plain, cache_dir / "fixed", depth, args.chunk_size, args.chunk_overlap),
        "semantic": lambda: build_semantic(plain, cache_dir / "semantic", depth, Settings.embed_model,
                                           args.semantic_threshold),
        "graphrag": lambda: build_graphrag(plain, cache_dir / "graphrag", depth,
                                           make_llm(args.graph_model, args.api_base, None), Settings.embed_model,
                                           args.chunk_size, args.graph_max_paths),
        # MinerU runs only when CA-RAG is evaluated
        "carag": lambda: build_carag(read_mineru(args), cache_dir / "carag", depth, args.parse_model,
                                     args.max_hops, args.carag_seed_k),
    }
    systems = {}
    for name in args.systems:
        logger.info(f"Building system: {name}")
        systems[name] = builders[name]()
    return systems


def _source_text(hit) -> str:
    """The corpus text of a retrieved node: the clause for CA-RAG, the chunk otherwise."""
    return hit.node.metadata.get("content") or hit.node.get_content()


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
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model, device="cuda", trust_remote_code=True)
    Settings.llm = MockLLM()

    # The baselines read the PDFs directly; CA-RAG reads them through MinerU when built
    plain = read_plain(args)

    # Load the QA set, synthesising it from the plain text when absent
    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        logger.info(f"Synthesising {args.num_questions} questions with {args.qa_model}")
        save_json(generate_qa(plain, make_llm(args.qa_model, args.api_base, None), args.num_questions), qa_path)
    qa = load_questions(qa_path)
    dropped = sum(len(item["evidence"]) - len(item["gold"]) for item in qa)
    logger.info(f"{len(qa)} questions from {qa_path}; {dropped} evidence without text left out of the gold")

    depth = max(args.retrieval_k, args.generation_k)
    systems = build_systems(args, plain, depth)

    # Retrieval: one ranking per question and system, reused by generation
    metrics = RetrievalMetrics(threshold=args.match_threshold)
    rankings, retrieval_rows, retrieval_table = {}, [], {}
    for name, retriever in systems.items():
        metrics.reset()
        for item in tqdm(qa, desc=f"retrieve:{name}"):
            hits = retriever.retrieve(item["question"])[:depth]
            print("question:")
            print(item["question"])
            print("\ngold:")
            print("\n\n".join(item["gold"]))
            print("\nhit: ")
            print("\n\n".join(
                h.text.replace("\\n", " ").replace("\r\n", " ").replace("\n", " ")
                for h in hits
            ))
            print('============================================')
            rankings[name, item["id"]] = hits
            scores = metrics.update([_source_text(h) for h in hits[:args.retrieval_k]], item["gold"])
            retrieval_rows.append({"system": name, "id": item["id"], "comments": item.get("comments", ""), **scores})
        retrieval_table[name] = metrics.compute()
        print(f"\n=== Retrieval: {name} ===\n{metrics.summarize()}")
    pd.DataFrame(retrieval_rows).to_csv(work_dir / "retrieval_per_question.csv", index=False)
    pd.DataFrame.from_dict(retrieval_table, orient="index").rename_axis("system").to_csv(work_dir / "retrieval.csv")

    if args.skip_generation:
        return

    # Generation: every answering LLM over every system's top nodes. Every question
    # is answered and cached; only those with a reference answer are scored
    metrics = GenerationMetrics()
    scored = sum(bool((item.get("answer") or "").strip()) for item in qa)
    logger.info(f"{scored}/{len(qa)} questions have a reference answer and are scored")
    generation_rows, generation_table = [], {}
    for model in args.llm_models:
        llm = make_llm(model, args.api_base, None)
        for name in systems:
            metrics.reset()
            # Answers are cached per (system, model) so an interrupted run resumes
            cache_path = work_dir / "answers" / f"{name}__{_slug(model)}.json"
            cached = {row["id"]: row for row in load_json(cache_path)} if cache_path.exists() else {}
            for item in tqdm(qa, desc=f"answer:{name}:{model}"):
                # ids are positions in the QA file, so a cached answer is reused only for the same question
                if cached.get(item["id"], {}).get("question") != item["question"]:
                    try:
                        prediction = answer(llm, item["question"], rankings[name, item["id"]][:args.generation_k])
                    except Exception as exc:
                        logger.warning(f"{model} failed on {item['id']}: {exc}")
                        continue
                    cached[item["id"]] = {"id": item["id"], "question": item["question"], "prediction": prediction}
                    save_json(list(cached.values()), cache_path)
                prediction = cached[item["id"]]["prediction"]
                if not (item.get("answer") or "").strip():
                    continue
                scores = metrics.update(prediction, item["answer"])
                generation_rows.append({"llm": model, "system": name, "id": item["id"], **scores})
            if scored:
                generation_table[model, name] = metrics.compute()
                print(f"\n=== Generation: {model} / {name} ===\n{metrics.summarize()}")
    if not generation_rows:
        print(f"\nNo reference answers yet; answers are cached under {work_dir / 'answers'} for scoring later")
        return
    pd.DataFrame(generation_rows).to_csv(work_dir / "generation_per_question.csv", index=False)
    table = pd.DataFrame.from_dict(generation_table, orient="index")
    table.index.names = ["llm", "system"]
    table.to_csv(work_dir / "generation.csv")


if __name__ == "__main__":
    main()
