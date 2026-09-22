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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import IdFuncCallable, build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import Field

# --- MinerU block types -------------------------------------------------------

# blocks that carry the text of the document
_TITLE_TYPES = {"doc_title", "paragraph_title", "title"}
_TEXT_TYPES = {"text", "paragraph", "list", "text_list", "code", "algorithm"}
_TABLE_TYPES = {"table", "simple_table", "complex_table"}
# page furniture and navigation aids; MinerU labels these for us
_SKIP_TYPES = {
    "page_number",
    "page_header",
    "page_footer",
    "page_aside_text",
    "header",
    "footer",
    "index",
    "image",
    "chart",
    "equation_interline",
    "reference_list",
}

# --- heading patterns ---------------------------------------------------------

# Containers name a span of the document rather than a requirement. Their depth
# is taken from the keyword because MinerU's own ``level`` is not reliable: the
# same "Subsection B11" is reported as doc_title/level=1 on one page and
# paragraph_title/level=2 on the next.
_CONTAINER_DEPTHS = {
    "part": 0,
    "chapter": 1,
    "schedule": 1,
    "annex": 1,
    "appendix": 1,
    "division": 2,
    "section": 3,
    "subsection": 4,
}
_CONTAINER_RE = re.compile(
    r"^(Part|Chapter|Schedule|Annex|Appendix|Division|Section|Subsection)\s+"
    r"([0-9A-Z]+(?:\.[0-9A-Z]+)*)\s*(?:[-–—:]\s*(.*))?$",
    re.I,
)
_CONTAINER_CN_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十]+)\s*(篇|章|节|節)\s*(.*)$")
_CN_DEPTHS = {"篇": 0, "章": 1, "节": 3, "節": 3}

# A heading that opens a single normative clause. Applied only to blocks MinerU
# has already called a title, so a numbered list item inside a paragraph cannot
# be mistaken for a clause of its own.
_CLAUSE_HEAD_RES = [
    # "Clause B7.1", "Regulation 12", "Article 4", "Rule 3A"
    re.compile(
        r"^(?:Clause|Regulation|Article|Rule|Paragraph)\s+"
        r"(?P<num>[0-9A-Z]+(?:\.[0-9A-Z]+)*)\s*[.:]?\s*(?P<head>.*)$",
        re.I,
    ),
    # "第 4.2.1 条" / "第 12 條"
    re.compile(r"^第\s*(?P<num>[0-9A-Z]+(?:\.[0-9A-Z]+)*)\s*[条條]\s*(?P<head>.*)$"),
    # "4.2.1 text" -- decimal numbering of CN/EU codes (at least two levels)
    re.compile(r"^(?P<num>\d+(?:\.\d+){1,3})\s*(?P<head>.*)$"),
    # "6. Magistrate may make orders" -- HK ordinance sections
    re.compile(r"^(?P<num>\d+[A-Z]?)\.\s+(?P<head>[A-Z一-鿿]\S*.*)$"),
]

# cross-references inside the clause text
# a reference number always contains a digit, which keeps ordinary words that
# follow "Table"/"Section" out of the reference list
_REF_NUM = r"([0-9]+[0-9A-Z]*(?:\.[0-9A-Z]+)*|[A-Z][0-9]+(?:\.[0-9A-Z]+)*)"
# citations of a clause, resolved inside the scope of the citing clause
_REF_RES = [
    re.compile(
        r"(?:Clause|Regulation|Article|Section|Subsection|Table|Figure)s?\s+" + _REF_NUM
    ),
    re.compile(r"第\s*" + _REF_NUM + r"\s*[条條章节節]"),
    re.compile(r"(?:表|图|圖)\s*" + _REF_NUM),
]
# Citations of a whole schedule or annex ("the requirements in Schedule 3") are
# not collected: they name a container of clauses rather than a single clause.

# document code at the beginning of a file name, e.g. "GB 50140-2005", "Cap 572",
# "BS 9999", "EN 1366-1", "GB/T 4968"
_DOC_CODE_RE = re.compile(
    r"^(?:(?:GB(?:/T)?|JGJ|CJJ|DIN|EN|BS|ISO|NFPA|Cap\.?)\s*[0-9]+[0-9A-Z]*"
    r"(?:[-–][0-9]+)?)",
    re.I,
)


def _doc_code(file_name: str) -> str:
    """Short document key used as the prefix of every clause ID."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", file_name).strip()
    matched = _DOC_CODE_RE.match(stem)
    return matched.group(0).strip() if matched else stem


_TAG_RE = re.compile(r"<[^>]+>")
_CELL_END_RE = re.compile(r"</t[dh]>", re.I)
_ROW_END_RE = re.compile(r"</tr>", re.I)


def _table_text(block: Dict[str, Any]) -> str:
    """Flatten a table into its caption and cell text.

    The HTML markup itself carries no meaning for retrieval and, being far
    longer than the text it wraps, would both dilute the embedding and push the
    clause over the length at which it gets split -- mid-tag, at that. The
    caption is kept because it is how the documents cite the table ("Table B1").
    """
    parts = [str(caption.get("content") or "").strip() for caption in block.get("captions") or []]
    html = str(block.get("content") or "")
    # separate the cells before dropping the tags, or the values run together
    text = _ROW_END_RE.sub("\n", _CELL_END_RE.sub(" | ", html))
    rows = [" ".join(_TAG_RE.sub(" ", row).split()).strip(" |") for row in text.splitlines()]
    parts.extend(row for row in rows if row)
    return " ".join(part for part in parts if part)


# TODO: 改成根据 code 自动确定
def _match_container(title: str) -> Optional[Tuple[int, str]]:
    """Return (depth, label) if the heading names a container, not a clause."""
    matched = _CONTAINER_RE.match(title)
    if matched:
        kind, num, rest = matched.group(1), matched.group(2), matched.group(3)
        depth = _CONTAINER_DEPTHS[kind.lower()]
        label = f"{kind.title()} {num}"
        return depth, f"{label} {rest.strip()}".strip() if rest else label
    matched = _CONTAINER_CN_RE.match(title)
    if matched:
        num, kind, rest = matched.group(1), matched.group(2), matched.group(3)
        label = f"第{num}{kind}"
        return _CN_DEPTHS[kind], f"{label} {rest.strip()}".strip()
    return None


def _match_clause_head(title: str) -> Optional[Tuple[str, str]]:
    """Return (number, remaining heading text) if the heading opens a clause."""
    for pattern in _CLAUSE_HEAD_RES:
        matched = pattern.match(title)
        if matched:
            return matched.group("num"), matched.group("head").strip()
    return None


def _extract_refs(text: str, doc_code: str, hier: List[str], own_id: str) -> List[str]:
    """Collect the clause identifiers cited inside ``text``.

    A citation is resolved in the scope of the citing clause, which is how the
    documents themselves read: "section 2" inside Schedule 3 means section 2 of
    that schedule.
    """
    refs: List[str] = []
    for pattern in _REF_RES:
        for num in pattern.findall(text):
            ref_id = _clause_id(doc_code, hier, num)
            if ref_id != own_id and ref_id not in refs:
                refs.append(ref_id)
    return refs


_SCOPE_RE = re.compile(r"^(Schedule|Annex|Appendix)\s+([0-9A-Z]+)", re.I)


def _clause_id(doc_code: str, hier: List[str], num: str) -> str:
    """Clause key, scoped by the schedule, annex or part that numbers it.

    Ordinances restart the numbering in every schedule, so the scope has to be
    part of the key: "Cap 572-Schedule 3-2".
    """
    for label in hier:
        matched = _SCOPE_RE.match(label)
        if matched:
            return f"{doc_code}-{matched.group(1).title()} {matched.group(2)}-{num}"
    return f"{doc_code}-{num}"


def _topic(head: str, hier: List[str]) -> str:
    """Functional subject of a clause.

    Ordinances and standards title their clauses ("6. Magistrate may make fire
    safety compliance orders"), codes of practice do not; there the innermost
    ancestor heading names the subject.
    """
    is_title = bool(head) and len(head) <= 100 and not head.endswith((".", ";", "；", "。", ","))
    if is_title:
        return head
    return hier[-1] if hier else ""


class ClauseNodeParser(NodeParser):
    """
    Split regulatory documents into one node per normative clause.

    Every node carries the schema ``<ID, Topic, Hier, Content, Ref>`` in its
    metadata; only ``Content`` is embedded and returned to the LLM, the rest is
    kept for retrieval and for reference expansion.
    """

    min_clause_chars: int = Field(
        default=40, description="Shorter clauses are merged into the previous one."
    )
    max_clause_chars: int = Field(
        default=4000, description="Longer clauses are cut into successive parts."
    )

    @classmethod
    def class_name(cls) -> str:
        return "ClauseNodeParser"

    @classmethod
    def from_defaults(
        cls,    
        min_clause_chars: int = 40,
        max_clause_chars: int = 4000,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[IdFuncCallable] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ) -> "ClauseNodeParser":
        """Initialize with parameters."""
        return cls(
            min_clause_chars=min_clause_chars,
            max_clause_chars=max_clause_chars,
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
        """
        The MinerU blocks of a document, in reading order.

        ``carag_build`` stores them on the document so that the expensive layout
        analysis is done once and cached; a document without them is parsed as a
        single untyped text block.
        """
        pages = doc.metadata.get("mineru_pages")
        if not pages:
            content = doc.get_content(metadata_mode=MetadataMode.NONE)
            return [{"type": "text", "content": content}] if content.strip() else []
        return [block for page in pages for block in page.get("blocks", [])]

    def _parse_document(self, doc: BaseNode) -> List[BaseNode]:
        file_name = doc.metadata.get("file_name", doc.node_id)
        doc_code = doc.metadata.get("doc_code") or _doc_code(file_name)

        hier: List[Tuple[int, str]] = []  # stack of (depth, label)
        clauses: List[Dict[str, Any]] = []

        for block in self._blocks(doc):
            kind = block.get("type", "")

            # Skip unnecessary blocks
            if kind in _SKIP_TYPES:
                continue

            # Tables always belong to the clause that precedes them
            if kind in _TABLE_TYPES:
                table = _table_text(block)
                if table and clauses:
                    clauses[-1]["lines"].append(table)
                continue

            # Clean the whitespace and skip empty blocks
            text = " ".join(str(block.get("content") or "").split())
            if not text:
                continue

            # Headings open a new clause, or a new container of clauses, or both
            if kind in _TITLE_TYPES:
                container = _match_container(text)
                if container is not None:
                    depth, label = container
                    while hier and hier[-1][0] >= depth:
                        hier.pop()
                    hier.append((depth, label))
                    continue

                # TODO: 缺少当 head 为空时用 LLM 补上
                clause = _match_clause_head(text)
                if clause is not None:
                    num, head = clause
                    clauses.append(
                        {
                            "num": num,
                            "head": head,
                            "hier": [label for _, label in hier],
                            "lines": [head] if head else [],
                        }
                    )
                    continue

                # An unnumbered heading ("Commentary") qualifies the clause it
                # follows, so it stays with that clause instead of opening one.
                if clauses:
                    clauses[-1]["lines"].append(text)
                continue

            # Text blocks are added to the current clause
            if kind in _TEXT_TYPES and clauses:
                clauses[-1]["lines"].append(text)

        return self._build_nodes(clauses, doc, doc_code)

    def _build_nodes(
        self, clauses: List[Dict[str, Any]], doc: BaseNode, doc_code: str
    ) -> List[BaseNode]:
        nodes: List[BaseNode] = []
        taken: set[str] = set()
        for clause in clauses:
            content = " ".join(clause["lines"]).strip()

            # a fragment too short to stand on its own belongs to the clause before
            if len(content) < self.min_clause_chars and nodes:
                previous = nodes[-1]
                previous.set_content(
                    f"{previous.get_content(metadata_mode=MetadataMode.NONE)} "
                    f"{clause['num']} {content}".strip()
                )
                continue

            if not content:
                continue

            clause_id = _clause_id(doc_code, clause["hier"], clause["num"])
            # a few documents restart their numbering per part, so the same key
            # can come round again; the ancestor path tells the repeats apart
            if clause_id in taken:
                scope = clause["hier"][0] if clause["hier"] else "cont"
                candidate, n = f"{clause_id} ({scope})", 1
                while candidate in taken:
                    n += 1
                    candidate = f"{clause_id} ({scope} {n})"
                clause_id = candidate
            taken.add(clause_id)

            metadata = {
                "ID": clause_id,
                "Topic": _topic(clause["head"], clause["hier"]),
                "Hier": " > ".join([doc_code, *clause["hier"]]),
                "Ref": _extract_refs(content, doc_code, clause["hier"], clause_id),
            }
            parts = self._split_long(content)
            for i, part in enumerate(parts):
                part_meta = dict(metadata)
                if len(parts) > 1:  # keep the key unique across the parts
                    part_meta["ID"] = f"{clause_id}({i + 1})"
                nodes.append(self._make_node(part, part_meta, doc))
        return nodes

    def _split_long(self, content: str) -> List[str]:
        """Cut an over-long clause on sentence boundaries."""
        if len(content) <= self.max_clause_chars:
            return [content]
        parts, current = [], ""
        for sentence in re.split(r"(?<=[.;。；])\s+", content):
            if current and len(current) + len(sentence) > self.max_clause_chars:
                parts.append(current.strip())
                current = ""
            current += sentence + " "
        if current.strip():
            parts.append(current.strip())
        return parts

    def _make_node(
        self, content: str, metadata: Dict[str, Any], doc: BaseNode
    ) -> TextNode:
        node = build_nodes_from_splits([content], doc, id_func=self.id_func)[0]
        node.metadata.update(metadata)
        # the reference list is bookkeeping, it is neither embedded nor shown
        node.excluded_embed_metadata_keys.append("Ref")
        node.excluded_llm_metadata_keys.append("Ref")
        return node
