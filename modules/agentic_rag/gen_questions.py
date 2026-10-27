from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You write evaluation questions for a retrieval system over building and fire safety regulations.
Given the evidence passages, write exactly ONE question that a building inspector might ask
and that can only be answered fully by reading ALL the passages together.

Rules:
1. The question must make sense on its own: never say "this clause", "the passage" or "the above".
2. Do not quote clause numbers or document identifiers in the question.
3. Ask about the substance (conditions, limits, requirements, procedures), in one sentence.

Return exactly ONE valid JSON object, without Markdown fences or explanatory text:
{"question": "..."}
"""

USER_PROMPT = """
Evidence:
{evidence}

Annotator note: {comments}

Write the question.
"""


class Question(BaseModel):
    question: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Write a question for each QA item from its evidence with a local LLM.")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/data/RAG/questions.json",
        help="JSON array of items with question, answer, evidence ([{source, text}]) and comments."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/dszh/workspace/tmp-inspector/data/RAG/test_questions_output.json",
        help="Where the items with the generated questions are written; the input is left untouched."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face text-generation model, the same one the clause parser uses for topics."
    )
    return parser.parse_args()


def format_evidence(evidence: list[dict]) -> str:
    """One numbered passage per evidence; an external document without text is given by its title."""
    lines = []
    for i, ev in enumerate(evidence, start=1):
        text = (ev.get("text") or "").strip()
        lines.append(f"[{i}] ({ev.get('source', '')}) {text}" if text else f"[{i}] Referenced document: {ev.get('source', '')}")
    return "\n\n".join(lines)


def generate_question(item: dict, model_pipeline) -> str:
    """Ask the model for one question, greedy as in the clause parser; empty when the output is not JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(
            evidence=format_evidence(item["evidence"]),
            comments=item.get("comments") or "none",
        )},
    ]
    raw = model_pipeline(messages, max_new_tokens=256, do_sample=False, return_full_text=False)[0]["generated_text"]
    try:
        return Question.model_validate(json.loads(raw)).question.strip()
    except Exception:
        logger.warning(f"Unparsable output, kept empty: {raw!r}")
        return ""


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    items = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logger.info(f"{len(items)} items from {args.input}")

    from transformers import pipeline
    model_pipeline = pipeline("text-generation", model=args.model, device_map="auto", max_length=None)

    for item in tqdm(items, desc="Generating questions"):
        item["question"] = generate_question(item, model_pipeline)
        item["answer"] = ""
        logger.info(f"{item['evidence'][0].get('source', '')}: {item['question']}")

    output = Path(args.output)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    logger.info(f"Written to {output}")


if __name__ == "__main__":
    main()
