from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Pattern

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import IdFuncCallable, build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import BaseModel, Field


# --- Clause numbering patterns ------------------------------------------------

@dataclass
class ClauseProfile:
    """Regular expressions for the headings and references of a document class."""

    section: Pattern[str]
    subsection: Pattern[str]
    subsubsection: Pattern[str]
    paragraph: Pattern[str]
    subparagraph: Pattern[str]
    ref: Pattern[str]

    @property
    def patterns(self) -> List[Pattern[str]]:
        return [
            self.section,
            self.subsection,
            self.subsubsection,
            self.paragraph,
            self.subparagraph,
        ]


_CLASS_PROFILE = {
    "GB": ClauseProfile(
        section=re.compile(r"^第\s*\d+\s*[章节篇](?:\s|$)|^\d+\s+"),
        subsection=re.compile(r"^\d+\.\d+(?:\s|$)"),
        subsubsection=re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
        paragraph=re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        subparagraph=re.compile(r"^\d+\.\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ref=re.compile(r"(?:第\s*)?\d+(?:\.\d+)*\s*[条章节節]"),
    ),
    "ISO": ClauseProfile(
        section=re.compile(r"^\d+(?:\s|$)"),
        subsection=re.compile(r"^\d+\.\d+(?:\s|$)"),
        subsubsection=re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
        paragraph=re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        subparagraph=re.compile(r"^\d+\.\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ref=re.compile(r"(?:Clause|Section)\s+\d+(?:\.\d+)*", re.I),
    ),
    "BS": ClauseProfile(
        section=re.compile(r"^Section\s+\d+(?:\s|$)", re.I),
        subsection=re.compile(r"^(?:Clause\s+\d+(?:\.\d+)*|\d+\.\d+)(?:\s|$)", re.I),
        subsubsection=re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
        paragraph=re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        subparagraph=re.compile(r"^\d+\.\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ref=re.compile(r"(?:Clause|Section)\s+\d+(?:\.\d+)*", re.I),
    ),
    "CAP": ClauseProfile(
        section=re.compile(r"^(?:Regulation|Article|Section)\s+\d+(?:\s|$)", re.I),
        subsection=re.compile(r"^(?:Clause\s+\d+(?:\.\d+)*|\d+\.\d+)(?:\s|$)", re.I),
        subsubsection=re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
        paragraph=re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        subparagraph=re.compile(r"^\d+\.\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ref=re.compile(r"(?:Regulation|Article|Section|Clause)\s+\d+(?:\.\d+)*", re.I),
    ),
    "COP": ClauseProfile(
        section=re.compile(r"^Part\s+(?:[A-Z]+|[IVXLCDM]+)(?:\s|$)", re.I),
        subsection=re.compile(r"^(?:Section\s+\d+(?:\.\d+)?|\d+\.\d+)(?:\s|$)", re.I),
        subsubsection=re.compile(r"^(?:Subsection\s+[A-Z]\d+|\d+\.\d+\.\d+)(?:\s|$)", re.I),
        paragraph=re.compile(r"^(?:Clause\s+[A-Z]\d+\.\d+|\d+\.\d+\.\d+\.\d+)(?:\s|$)", re.I),
        subparagraph=re.compile(r"^\d+\.\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ref=re.compile(r"(?:Regulation|Article|Section|Clause)\s+\d+(?:\.\d+)*", re.I),
    ),
}


def _doc_class(doc_code: Optional[str]) -> str:
    """Return the parser profile name for a corpus document code."""
    code = (doc_code or "").strip().upper()
    if code.startswith("CAP"):
        return "CAP"
    if code.startswith("COP"):
        return "COP"
    if code.startswith("BS"):
        return "BS"
    if code.startswith("ISO"):
        return "ISO"
    if code.startswith(("GB")):
        return "GB"
    return "ISO"


def _profile(doc_code: Optional[str]) -> ClauseProfile:
    return _CLASS_PROFILE.get(_doc_class(doc_code), _CLASS_PROFILE["ISO"])


# --- MinerU block types -------------------------------------------------------

_TITLE_TYPES = {
    "header", 
    "doc_title", 
    "paragraph_title", 
    "title"
}
_TEXT_TYPES = {
    "text", 
    "paragraph", 
    "list", 
    "text_list", 
    "code", 
    "algorithm"
}
_TABLE_TYPES = {
    "table", 
    "simple_table", 
    "complex_table"
}
_SKIP_TYPES = {
    "page_number",
    "page_header",
    "page_footer",
    "page_aside_text",
    "footer",
    "index",
    "image",
    "chart",
    "equation_interline",
    "reference_list",
}


# --- MinerU block text processing ---------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_CELL_END_RE = re.compile(r"</t[dh]>", re.I)
_ROW_END_RE = re.compile(r"</tr>", re.I)


def _table_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the text of a table block."""
    parts = [str(caption.get("content") or "").strip() for caption in block.get("captions") or []]
    html = str(block.get("content") or "")
    
    # separate the cells before dropping the tags, or the values run together
    text = _ROW_END_RE.sub("\n", _CELL_END_RE.sub(" | ", html))
    rows = [
        " ".join(_TAG_RE.sub(" ", row).split()).strip(" |") 
        for row in text.splitlines()
    ]
    parts.extend(row for row in rows if row)
    
    text = " ".join(part for part in parts if part)
    return text, -1


def _main_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the main text of a block."""
    text = " ".join(str(block.get("content") or "").split())
    return text, -1


def _head_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the heading text of a block."""
    text = " ".join(str(block.get("content") or "").split())
    profile = _profile(doc_code)

    for depth, pattern in enumerate(profile.patterns[:4], start=1):
        if pattern.match(text):
            return text, depth

    return text, -1


def _head_label(text: str, doc_code: Optional[str]) -> str:
    """The bare label of a heading, in the form of ``local_cites``.

    "Clause B6.1 Single staircase" -> "B6.1", "Part C" -> "C", "第 4 章" -> "4".
    """
    for pattern in _profile(doc_code).patterns:
        match = pattern.match(text)
        if match:
            words = match.group(0).replace("第", " ").strip(" 章节篇条").split()
            return words[-1] if words else ""
    return ""


# --- LLM Extraction -----------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = """
You are an expert in extracting structured data from regulations and codes.
Your task is to analyze the provided clause text and return exactly ONE valid JSON object.
Do NOT use Markdown formatting (no ```json fences), and do NOT include any explanatory text.

The JSON object must have exactly these three keys:
{
    "topic": "A concise, one-sentence summary of the clause's main subject.",
    "local_cites": ["Labels of the clauses of the current document cited by the text"],
    "external_cites": ["Identifiers of the other documents cited by the text"]
}

The identifier of the current document is given before the clause text. Sort every citation by these rules:

1. local_cites: a clause, subsection, section or part cited without naming another document, or naming the current document ("this Code", "this Ordinance", its identifier).
   Write only its label, dropping words such as Clause, Paragraph, Subsection, Section, Part, Regulation, Article, 第, 条, 章, 节:
   - "Clause B6.1" -> "B6.1"
   - "Subsection B6" -> "B6"
   - "Section 5 of this Code" -> "5"
   - "Part C" -> "C"
   - "paragraphs 4.2.1 and 4.2.2" -> "4.2.1", "4.2.2"
   - "第 4.2 条" -> "4.2"
2. external_cites: another standard, code, ordinance or regulation.
   Write only its identifier, keeping the part number and the year exactly as printed, and dropping its title and any clause of it:
   - "BS EN ISO 1182:2010, Reaction to fire tests for products – Non-combustibility test" -> "BS EN ISO 1182:2010"
   - "BS EN 12101-1:2005 Smoke and heat control systems" -> "BS EN 12101-1:2005"
   - "Clause 5 of BS 5588-1:1990" -> "BS 5588-1:1990"
   - "ISO 7240-14" -> "ISO 7240-14"
   - "GB 55036-2023 消防设施通用规范" -> "GB 55036-2023"
3. A Hong Kong ordinance or regulation is written as "Cap" and its chapter number, however the text spells it:
   - "Buildings Ordinance (Cap.123)" -> "Cap 123"
   - "Cap. 95F" -> "Cap 95F"
   - "Fire Safety (Buildings) Ordinance, Cap 572" -> "Cap 572"
4. Leave out tables, figures, diagrams, annexes and the clause itself. List each citation once; use an empty list when there is none.
"""

DEFAULT_USER_PROMPT = """
Clause text:
{content}

Extract the topic, the local_cites and the external_cites.
"""


class ClauseExtraction(BaseModel):
    topic: str = ""
    local_cites: list[str] = Field(default_factory=list)
    external_cites: list[str] = Field(default_factory=list)


def _extract_clause(
    content: str,
    model_pipeline: Any,
    user_prompt: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Extract topic and references with the configured local model."""

    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(content=content)}
    ]

    # Generate the text with the model
    gen_text = model_pipeline(
        messages,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False
    )
    raw = gen_text[0].get("generated_text", "")

    try:
        result = ClauseExtraction.model_validate(json.loads(raw))
    except Exception as e:
        result = ClauseExtraction()

    return result.model_dump()


# --- Clause-aware Node Parser -------------------------------------------------

class ClauseNodeParser(NodeParser):
    """
    Split regulatory documents into one node per normative clause.

    Every node carries ``topic``, ``hier``, ``labels``, ``local_cites`` and ``external_cites`` in
    its metadata; only clause content is embedded and returned to the LLM.
    """

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    user_prompt: str = DEFAULT_USER_PROMPT
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def class_name(cls) -> str:
        return "ClauseNodeParser"

    @classmethod
    def from_defaults(
        cls,    
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[IdFuncCallable] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ) -> "ClauseNodeParser":
        """Initialize with parameters."""
        return cls(
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            callback_manager=callback_manager or CallbackManager([]),
            id_func=id_func or default_id_func,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        # Initialize the model pipeline for clause extraction
        from transformers import pipeline
        model_pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            max_length=None,
        )

        # Parse each document into clauses and build nodes
        try:
            for node in nodes_with_progress:
                all_nodes.extend(self._parse_document(node, model_pipeline))
            return all_nodes

        # Release pipeline resources and clear GPU memory after processing
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _blocks(doc: BaseNode) -> Iterable[Dict[str, Any]]:
        """The MinerU blocks of a document, in reading order."""
        # Get the MinerU page list from the document metadata
        pages = doc.metadata.get("mineru_pages")

        # Single-page documents
        if not pages:
            content = doc.get_content(metadata_mode=MetadataMode.NONE)
            return [{"type": "text", "content": content}] if content.strip() else []

        # Multi-page documents
        return [block for page in pages for block in page.get("blocks", [])]

    def _parse_document(self, doc: BaseNode, model_pipeline: Any) -> List[BaseNode]:
        """Parse a single document into one node per clause."""
        doc_code = str(doc.metadata.get("code") or "").strip()

        hier: List[Tuple[int, str]] = [(0, doc_code)] if doc_code else []
        # Labels of the open headings, which a clause is cited by
        head_labels: List[Tuple[int, str]] = []
        clauses: List[Dict[str, Any]] = []

        for block in self._blocks(doc):
            # Get the type of the block
            block_type = block.get("type", "")

            # Skip unnecessary blocks
            if block_type in _SKIP_TYPES:
                continue

            # Tables always belong to the clause that precedes them
            if block_type in _TABLE_TYPES:
                text, depth = _table_text(block, doc_code)
                if text and clauses:
                    clauses[-1]["lines"].append(text)
                continue
            
            # Text blocks are added to the current clause
            if block_type in _TEXT_TYPES and clauses:
                text, depth = _main_text(block, doc_code)
                if text and clauses:
                    clauses[-1]["lines"].append(text)
                continue

            # Headings open a new clause, or a new container of clauses, or both
            if block_type in _TITLE_TYPES:
                text, depth = _head_text(block, doc_code)

                if depth > 0 and depth <= 4:
                    # Stack the heading in the hierarchy, popping any that are deeper than it
                    while hier and hier[-1][0] >= depth:
                        hier.pop()
                    hier.append((depth, text))
                    head_labels = [(d, l) for d, l in head_labels if d < depth]
                    head_labels.append((depth, _head_label(text, doc_code)))

                    clauses.append(
                        {
                            "head": text,
                            "hier": [label for _, label in hier],
                            "labels": [label for _, label in head_labels if label],
                            "lines": [text],
                        }
                    )
                else:
                    if clauses:
                        clauses[-1]["lines"].append(text)
                continue

        return self._build_nodes(clauses, doc, model_pipeline)

    def _build_nodes(
        self,
        clauses: List[Dict[str, Any]],
        doc: BaseNode,
        model_pipeline: Any,
    ) -> List[BaseNode]:
        """Build nodes from the parsed clauses."""
        # Build a node for each clause
        nodes: List[BaseNode] = []
        for clause in clauses:
            content = " ".join(clause["lines"]).strip()

            if not content:
                continue

            # Exttract topic and references with agentic LLM
            extracted = _extract_clause(
                content,
                model_pipeline,
                self.user_prompt,
                self.system_prompt,
            )

            print(f"{content} \n -> {extracted}\n")

            metadata = {
                "topic": extracted["topic"],
                "content": content,
                "hier": " > ".join(clause["hier"]),
                "labels": clause["labels"],
                "local_cites": extracted["local_cites"],
                "external_cites": extracted["external_cites"],
            }
            nodes.append(self._make_node(metadata, doc))

        return nodes

    def _make_node(
        self, metadata: Dict[str, Any], doc: BaseNode
    ) -> TextNode:
        """Make a node from the clause content and metadata."""
        
        # Make the node
        topic = str(metadata.get("topic") or "").strip()
        node = build_nodes_from_splits([topic], doc, id_func=self.id_func)[0]
        node.metadata.update(metadata)
        node.metadata.pop("mineru_pages", None)

        # Only topic should contribute to the embedding
        node.excluded_embed_metadata_keys.extend(
            [
                "content",
                "hier",
                "labels",
                "local_cites",
                "external_cites",
            ]
        )
        node.excluded_llm_metadata_keys.extend(
            ["labels", "local_cites", "external_cites"]
        )
        
        return node

