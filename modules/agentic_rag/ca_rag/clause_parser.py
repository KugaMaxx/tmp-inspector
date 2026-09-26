"""Clause-aware chunking of fire safety codes and regulations.

A regulatory document is split so that one node holds exactly one independent
normative clause, described by the tuple <ID, Topic, Hier, Content, Ref>.
Numbering styles of EU/UK codes (``Clause 4.2.1``, ``Regulation 12``), Chinese
codes (``4.2.1``, ``第 4.2.1 条``) and Hong Kong ordinances (``6. Heading``)
are all recognised.

The input is the block list of a MinerU parse, not raw text: MinerU already
labels every block with its role on the page, so page numbers, running headers
and contents entries arrive pre-separated and do not have to be guessed from
their shape. What MinerU does not provide is meaning -- the hierarchy depth of a
heading and the cross-references inside a clause are still derived here.
"""

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
    return text, 0


def _main_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the main text of a block."""
    text = " ".join(str(block.get("content") or "").split())
    return text, 0


def _head_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the heading text of a block."""
    text = " ".join(str(block.get("content") or "").split())
    profile = _profile(doc_code)

    for depth, pattern in enumerate(profile.patterns[:4], start=1):
        if pattern.match(text):
            return text, depth
        
    return text, 0


# --- LLM Extraction -----------------------------------------------------------

_MODEL_PIPELINES: Dict[str, Any] = {}


def _get_model_pipeline(model_name: str) -> Any:
    """Load one text-generation pipeline per model name and reuse it globally."""
    if model_name not in _MODEL_PIPELINES:
        from transformers import pipeline

        _MODEL_PIPELINES[model_name] = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_length=None,
        )
        
    return _MODEL_PIPELINES[model_name]


DEFAULT_SYSTEM_PROMPT = """You are an expert in extracting structured data from regulations and codes.
Your task is to analyze the provided clause text and return exactly ONE valid JSON object. 
Do NOT use Markdown formatting (no ```json fences), and do NOT include any explanatory text.

The JSON object must have exactly these three keys:
{
    "topic": "A concise, one-sentence summary of the clause's main subject.",
    "local_ref": ["List of clause identifiers that belong to the SAME document"],
    "global_ref": ["List of identifiers that belong to EXTERNAL standards or other documents"]
}

### Examples:
1. GLOBAL REFERENCE (`global_ref`):
- "BS EN 12101-1:2005 Smoke and heat control systems - Specification for smoke barriers"
- "ISO 7240-14:2013"
- "BS ISO 10294-1:1996, Fire-resistance tests"
- "Cap 572 Fire Safety (Buildings) Ordinance" / "Buildings Ordinance (Cap.123)" / Cap.95F
- "GB 55036-2023 消防设施通用规范"
2. LOCAL REFERENCE (`local_ref`):
- "Clause 4.2.1"
- "Section 5"
- "Subsection 3.1.2"
- "Part C"
- "Annex A"
- "Diagram 2"
- "第 4.2 条"
"""

DEFAULT_USER_PROMPT = """
Clause text:
{content}

Extract the topic and classify every cited reference as local_ref or global_ref."""


class ClauseExtraction(BaseModel):
    topic: str = ""
    local_ref: list[str] = Field(default_factory=list)
    global_ref: list[str] = Field(default_factory=list)


def _extract_clause(
    content: str,
    model_name: Optional[str],
    user_prompt: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Extract topic and references with the configured local model."""

    # Lazy load the model pipeline
    model_pipeline = _get_model_pipeline(model_name)

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

    print(f"{content} -> {raw}")

    try:
        result = ClauseExtraction.model_validate(json.loads(raw))
    except Exception as e:
        result = ClauseExtraction()

    return result.model_dump()


# --- Clause-aware Node Parser -------------------------------------------------

class ClauseNodeParser(NodeParser):
    """
    Split regulatory documents into one node per normative clause.

    Every node carries ``Topic``, ``Hier``, ``local_ref`` and ``global_ref`` in
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
        
        for node in nodes_with_progress:
            all_nodes.extend(self._parse_document(node))
        
        return all_nodes

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

    def _parse_document(self, doc: BaseNode) -> List[BaseNode]:
        """Parse a single document into one node per clause."""
        doc_code = doc.metadata.get("code")

        hier: List[Tuple[int, str]] = []
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

                    clauses.append(
                        {
                            "head": text,
                            "hier": [label for _, label in hier],
                            "lines": [text],
                        }
                    )
                else:
                    if clauses:
                        clauses[-1]["lines"].append(text)
                continue

        return self._build_nodes(clauses, doc, doc_code)

    def _build_nodes(
        self, clauses: List[Dict[str, Any]], doc: BaseNode, doc_code: str
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
                content, self.model_name, self.user_prompt, self.system_prompt
            )

            metadata = {
                "topic": extracted["topic"],
                "hier": " > ".join([doc_code, *clause["hier"]]),
                "local_ref": extracted["local_ref"],
                "global_ref": extracted["global_ref"],
            }
            nodes.append(self._make_node(content, metadata, doc))

        return nodes

    def _make_node(
        self, content: str, metadata: Dict[str, Any], doc: BaseNode
    ) -> TextNode:
        """Make a node from the clause content and metadata."""
        
        # Make the node with the content and metadata
        node = build_nodes_from_splits([content[:12000]], doc, id_func=self.id_func)[0]
        node.metadata.update(metadata)

        # Reference lists are bookkeeping.
        node.excluded_embed_metadata_keys.extend(["local_ref", "global_ref"])
        node.excluded_llm_metadata_keys.extend(["local_ref", "global_ref"])
        
        return node

