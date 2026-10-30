from __future__ import annotations

import re
import json
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Pattern

import pandas as pd

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import IdFuncCallable, build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, Document, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from pydantic import BaseModel, Field, field_validator


# --- Citation Extraction by LLM -----------------------------------------------

DEFAULT_SYSTEM_PROMPT = r"""
You are an expert in extracting structured data from regulations and codes. Your task is to analyze the provided clause text and return exactly ONE valid JSON object.

The JSON object must have exactly these two keys:
{
    "topic": "A concise, one-sentence summary of the clause's main subject.",
    "cites": [{"<code>": "<clause id, or empty string>"}]
}

Here are some reference <code> and corresponding <clause id> rules for your reference:
| code | name | rules |
| ---- | ---- | ----- |
| CoP-FSB | Code of Practice for Fire Safety in Buildings | ONLY matching the following regex patterns can be regarded as <clause id>: Part [A-Z] (e.g., "Part A"), Section \d+ (e.g., "Section 1", "Section 12"), Subsection [A-Z]\d+ (e.g., "Subsection A1", "Subsection B12"), Clause [A-Z]\d+\.\d+ (e.g., "Clause A1.1", "Clause B12.3"), and [A-Z]\d+\.\d+ (e.g., C12.1); however, MUST NOT be regarded as citations: Performance Requirement [A-Z]\d+ (such as "Performance Requirement B1"). |
| CoP-FSI | Code of Practice for Minimum Fire Service Installations and Equipment and Inspection, Testing and Maintenance of Installations and Equipment |  |
| Cap. 59V | Cap. 59V Factories and Industrial Undertakings (Fire Precautions in Notifiable Workplaces) Regulations |  |
| Cap. 95F | Cap. 95F Fire Services (Fire Hazard Abatement) Regulation |  |
| Cap. 123 | Cap. 123 Buildings Ordinance | ONLY text matching the regex pattern \d+[A-Z]* (such as "41" or "23A") can be regarded as <clause id>; if such a pattern is followed by parenthesized digits like \(\d+\) (e.g., "41(1)" or "23A(1)"), ONLY the base part preceding the parentheses (e.g., "41", "23A") must be regarded as the <clause id>. |
| Cap. 123F | Cap. 123F Building (Planning) Regulations |  |
| Cap. 123Q | Cap. 123Q Building (Construction) Regulations |  |
| Cap. 279A | Cap. 279A Education Regulations |  |
| Cap. 295 | Cap. 295 Dangerous Goods Ordinance |  |
| Cap. 509 | Cap. 509 Occupational Safety and Health Ordinance |  |
| BS EN 1634-3:2004 | BS EN 1634-3:2004 Fire resistance and smoke control tests for door and shutter assemblies, openable windows and elements of building hardware. Smoke control test for door and shutter assemblies |  |

## Some examples:
INPUT:  
"Clause B11.4 Where a flat has three or more levels, at least two levels should each have an access to a protected exit."
OUTPUT:
{
    "topic": "Level requirement for access to a protected exit in flats with three or more levels.",
    "cites": []
}
NOTE:
The label or number at the very start of the clause text is the heading of the clause itself, not a citation.


INPUT:
"Clause B10.1 In case of a building with one required staircase, attention is drawn to the requirements in Subsection B6."
OUTPUT:
{
    "topic": "Requirements for buildings with one required staircase",
    "cites": [{"CoP-FSB": "Subsection B6"}]
}
NOTE:
A clause, subsection, section or part cited without naming another document, or naming the current document ("this Code", "this Ordinance", its identifier), belongs to the current document.


INPUT:
"Clause B5.3 If an exit route leads to an open area at any upper floor levels of the building such as the podium level, instead of leading directly to an ultimate place of safety, the exit route at such floor level: (a) should be adequately defined by permanent features e.g. handrail or railing; (b) should lead to an ultimate place of safety; and (c) should be designed and constructed as if it is part of the protected exit complying with the requirements in this Part and Clause C12.4. See Diagram B1."
OUTPUT:
{
    "topic": "Requirements for exit routes leading to upper floor open areas instead of ultimate place of safety",
    "cites": [{"CoP-FSB": "Clause C12.4"}]
}
NOTE:
"this Part" is a self-reference and must NOT be extracted. "Clause C12.4" is a citation within the same document.


INPUT:
"Clause A8 Functional Statement for Means of Escape Building (Planning) Regulation 41(1): Means of escape. Every building shall be provided with such means of escape in case of emergency as may be required by the intended use of the building. Performance Requirement B1 A building, fire compartment or storey should be provided with adequate means of escape for all occupants to evacuate safely without being overwhelmed by the effects of fire. Performance Requirement B2 A building, fire compartment or storey should have adequate fire safety provisions to protect evacuating occupants from the impact of fire."
OUTPUT:
{
    "topic": "Functional statement for means of escape with performance requirements",
    "cites": [{"Cap. 123F": "41"}]
}
NOTE:
"Performance Requirement B1" and "Performance Requirement B2" are NOT citations — they are inline headings within the same clause. "Building (Planning) Regulation 41(1)" maps to Cap. 123F, and ONLY the base part "41" is extracted (parenthesized "(1)" is stripped per Cap. 123F rules).


INPUT:
"Clause A10 Functional statements for means of access Building (Planning) Regulation 41A: Access staircases for firemen Building (Planning) Regulation 41B: Fireman's lifts Building (Planning) Regulation 41C: Firefighting and rescue stairway Building (Planning) Regulation 41D: Emergency vehicular access Every building shall be provided with adequate access to allow firemen safe and unobstructed access to all floors of the building in the event of a fire. Every building shall also be provided with an emergency vehicular access to allow safe and unobstructed access of a vehicle of the Fire Services Department to the building and to provide for the safe operation of such a vehicle in the event of fire. Performance Requirement D1 A building should be provided with the following provisions to assist firefighting: (a) Access staircases for firemen; (b) Fireman's lifts; (c) Firefighting and rescue stairways; and (d) Emergency vehicular access."
OUTPUT:
{
    "topic": "Functional statements for means of access including staircases, lifts, stairways and vehicular access for firefighting",
    "cites": [{"Cap. 123F": "41A"}, {"Cap. 123F": "41B"}, {"Cap. 123F": "41C"}, {"Cap. 123F": "41D"}]
}
NOTE:
"Building (Planning) Regulation" is an alternative name for Cap. 123F. Each regulation number (41A, 41B, 41C, 41D) is a separate citation. "Performance Requirement D1" is NOT a citation. The clause heading "Clause A10" is NOT a citation.


INPUT:
"Clause B5.4 In a building where there is an area of special hazard that is directly associated with an area of normal hazard, any opening from such area of special hazard into any exit route should be provided with a protected lobby. The exit route from the area of normal hazard should not pass through any area of special hazard. [Note: A restaurant with kitchens or food preparation rooms using gas fuel or electricity for food production is considered as "notifiable workplace" under Factories and Industrial Undertakings (Fire Precautions in Notifiable Workplaces) Regulations (Cap.59V)."
OUTPUT:
{
    "topic": "Protected lobby requirements between special hazard and normal hazard areas",
    "cites": [{"Cap. 59V": ""}]
}
NOTE:
"Factories and Industrial Undertakings (Fire Precautions in Notifiable Workplaces) Regulations (Cap.59V)" is a reference to the regulation as a whole without a specific clause number, so clause id is empty string.


INPUT:
"Clause B5.5 Every part of an exit route should be provided at all time with lighting of a horizontal illuminance at floor level of not less than 30 lux. The lighting can be a combination of natural and artificial light and should be backed up by an emergency lighting system that complies with the Code of Practice for Minimum Fire Service Installations and Equipment."
OUTPUT:
{
    "topic": "Lighting requirements for exit routes including emergency lighting backup",
    "cites": [{"CoP-FSI": ""}]
}
NOTE:
"Code of Practice for Minimum Fire Service Installations and Equipment" is referenced by its full name without a specific section number, so clause id is empty string. The code is CoP-FSI.

INPUT:
"Clause B18.2 Every refuge floor, except that provided under Clauses B18.3 and B18.5, should comply with the following requirements: (a) There is no occupied accommodation or accessible mechanical plant room, except fire services water tanks and associated fire service installation plant room, at the same level as the refuge floor; (d) The area for refuge should be separated from the remainder of the building in accordance with the requirements in Part C; (g) Every part of the area for refuge should be provided with at all time with lighting of a horizontal illuminance at floor level of not less than 30 lux. The lighting can be a combination of natural and artificial light and should be backed up by an emergency lighting system that complies with the Code of Practice for Minimum Fire Service Installations and Equipment; (i) A refuge floor should be served by fireman's lift but not other lifts."
OUTPUT:
{
    "topic": "Requirements for refuge floors including separation, lighting, and lift access",
    "cites": [{"CoP-FSB": "Clause B18.3"}, {"CoP-FSB": "Clause B18.5"}, {"CoP-FSB": "Part C"}, {"CoP-FSI": ""}]
}
NOTE:
"Clauses B18.3 and B18.5" — two internal citations, extracted individually. "Part C" — internal reference to a part. "Code of Practice for Minimum Fire Service Installations and Equipment" — external reference (CoP-FSI) without specific section.
"""

DEFAULT_USER_PROMPT = """
Current document:
{code}

Clause text:
{content}

Extract the topic and the cites.
"""


# --- Clause numbering patterns ------------------------------------------------

class ClauseProfile:
    """Heading patterns of a document, one per level, outermost first, plus the
    caption patterns of its tables and diagrams, which sit outside the levels."""

    def __init__(
        self,
        levels: List[Pattern[str]],
        table: Optional[Pattern[str]] = None,
        diagram: Optional[Pattern[str]] = None,
    ) -> None:
        self.levels = levels
        self.table = table
        self.diagram = diagram


# Default clause heading patterns for each document type
_PROFILE: Dict[str, ClauseProfile] = {
    "ISO": ClauseProfile(
        [
            re.compile(r"^\d+(?:\s|$)"),
            re.compile(r"^\d+\.\d+(?:\s|$)"),
            re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
            re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ],
    ),
    "BS": ClauseProfile(
        [
            re.compile(r"^Section\s+\d+(?:\s|$)", re.I),
            re.compile(r"^(?:Clause\s+\d+(?:\.\d+)*|\d+\.\d+)(?:\s|$)", re.I),
            re.compile(r"^\d+\.\d+\.\d+(?:\s|$)"),
            re.compile(r"^\d+\.\d+\.\d+\.\d+(?:\s|$)"),
        ],
        table=re.compile(r"^Table\s+\d+(?:\.\d+)?(?:[\s:.]|$)", re.I),
    ),
    "CaP": ClauseProfile(
        [
            re.compile(r"^Part\s+(?:[A-Z]+|[IVXLCDM]+)(?:\s|$)", re.I),
            # "49.", and the sections inserted after it: "49A.", "5AA."
            re.compile(r"^\d+[A-Z]*\.(?:\s|$)"),
        ],
    ),
    "CoP-FSB": ClauseProfile(
        [
            re.compile(r"^Part\s+(?:[A-Z]+|[IVXLCDM]+)(?:\s|$)", re.I),
            re.compile(r"^(?:Section\s+\d+(?:\.\d+)?|\d+\.\d+)(?:\s|$)", re.I),
            re.compile(r"^(?:Subsection\s+[A-Z]\d+|\d+\.\d+\.\d+)(?:\s|$)", re.I),
            re.compile(r"^(?:Clause\s+[A-Z]\d+\.\d+|\d+\.\d+\.\d+\.\d+)(?:\s|$)", re.I),
        ],
        table=re.compile(r"^Table\s+[A-Z]\d+(?:\.\d+)?(?:[\s:.]|$)", re.I),
    ),
}
_DEFAULT_PROFILE = "ISO"


def _code_tokens(code: str) -> List[str]:
    """'Cap. 279A' -> ['CAP', '279A'], 'CoP-FSB' -> ['COP', 'FSB']."""
    return re.findall(r"[0-9A-Z]+", code.upper())


def _profile(doc_code: Optional[str]) -> ClauseProfile:
    """The profile whose key is the longest token prefix of the document code."""
    tokens = _code_tokens(doc_code or "")
    best, best_len = _DEFAULT_PROFILE, 0
    for key in _PROFILE:
        key_tokens = _code_tokens(key)
        if len(key_tokens) > best_len and tokens[:len(key_tokens)] == key_tokens:
            best, best_len = key, len(key_tokens)
    return _PROFILE[best]


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

def _extract_table_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, str]:
    """Process the text of a table block, and the table label in its caption."""
    parts = [str(caption.get("content") or "").strip() for caption in block.get("captions") or []]

    pattern = _profile(doc_code).table
    matches = [pattern.match(part) for part in parts] if pattern else []
    head = next((m.group(0).strip().rstrip(".:") for m in matches if m), "")

    # the caption on the first line, the table below it
    caption = " ".join(part for part in parts if part)
    html = str(block.get("content") or "")

    if "<table" not in html.lower():
        table = " ".join(re.compile(r"<[^>]+>").sub(" ", html).split())
    else:
        try:
            # keep every cell as written: no NaN for "NA" or empty cells, no "1,000" -> 1000
            df = pd.read_html(
                StringIO(html), flavor="lxml", header=None, keep_default_na=False, thousands=None
            )[0]
        except ValueError:
            # a table without rows
            df = pd.DataFrame()

        # the HTML table as a markdown table, merged cells repeated in each cell they cover
        lines = [
            "| " + " | ".join(" ".join(str(cell).split()).replace("|", "\\|") for cell in row) + " |"
            for row in df.values.tolist()
        ]
        if lines:
            lines.insert(1, "| " + " | ".join(["----"] * df.shape[1]) + " |")
        table = "\n".join(lines)

    text = "\n".join(part for part in (caption, table) if part)
    return text, head


def _extract_main_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int]:
    """Process the main text of a block."""
    text = " ".join(str(block.get("content") or "").split())
    return text, -1


def _extract_head_text(block: Dict[str, Any], doc_code: Optional[str]) -> Tuple[str, int, str]:
    """Process the heading text of a block."""
    text = " ".join(str(block.get("content") or "").split())

    for depth, pattern in enumerate(_profile(doc_code).levels, start=1):
        match = pattern.match(text)
        if match:
            return text, depth, match.group(0).strip().rstrip(".")

    return text, -1, ""


class ClauseExtraction(BaseModel):
    topic: str = ""
    # [{"CoP-FSI": "4.2.1"}, {"Cap. 95F": "45"}, {"BS 5588-1:1990": ""}]
    cites: List[Dict[str, str]] = Field(default_factory=list)

    @field_validator("cites", mode="before")
    @classmethod
    def _one_key_each(cls, value: Any) -> List[Dict[str, str]]:
        """Split multi-key objects into one citation each, dropping duplicates."""
        cites: List[Dict[str, str]] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            for doc, label in item.items():
                cite = {str(doc).strip(): str(label or "").strip()}
                if next(iter(cite)) and cite not in cites:
                    cites.append(cite)
        return cites


def _extract_clause(
    content: str,
    doc_code: str,
    model_pipeline: Any,
    user_prompt: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Extract topic and references with the configured local model."""

    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(code=doc_code, content=content)}
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
    """Split regulatory documents into one node per normative clause."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    user_prompt: str = DEFAULT_USER_PROMPT
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def class_name(cls) -> str:
        return "ClauseNodeParser"

    @classmethod
    def from_defaults(
        cls,    
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
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

    def _postprocess_parsed_nodes(
        self, nodes: List[BaseNode], parent_doc_map: Dict[str, Document]
    ) -> List[BaseNode]:
        nodes = super()._postprocess_parsed_nodes(nodes, parent_doc_map)

        drop = {"mineru_pages"}
        for node in nodes:
            for key in drop:
                node.metadata.pop(key, None)
            for rel in node.relationships.values():
                for info in rel if isinstance(rel, list) else [rel]:
                    if info.metadata and drop & info.metadata.keys():
                        info.metadata = {k: v for k, v in info.metadata.items() if k not in drop}
        return nodes

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

        # Open headings as (depth, matched heading), outermost first
        hier: List[Tuple[int, str]] = []
        clauses: List[Dict[str, Any]] = []
        # Labelled tables, kept apart so the text after them returns to the clause
        tables: List[Dict[str, Any]] = []

        for block in self._blocks(doc):
            # Get the type of the block
            block_type = block.get("type", "")

            # Skip unnecessary blocks
            if block_type in _SKIP_TYPES:
                continue

            # Labelled tables are nodes under the clause they sit in, the others its text
            if block_type in _TABLE_TYPES:
                text, head = _extract_table_text(block, doc_code)
                if text and head:
                    tables.append(
                        {
                            "head": text,
                            "hier": (clauses[-1]["hier"] if clauses else []) + [head],
                            "lines": [text],
                        }
                    )
                elif text and clauses:
                    clauses[-1]["lines"].append(text)
                continue
            
            # Text blocks are added to the current clause
            if block_type in _TEXT_TYPES and clauses:
                text, depth = _extract_main_text(block, doc_code)
                if text and clauses:
                    clauses[-1]["lines"].append(text)
                continue

            # Headings open a new clause, or a new container of clauses, or both
            if block_type in _TITLE_TYPES:
                text, depth, head = _extract_head_text(block, doc_code)

                if depth > 0:
                    # Stack the heading in the hierarchy, popping any that are deeper than it
                    while hier and hier[-1][0] >= depth:
                        hier.pop()
                    hier.append((depth, head))

                    clauses.append(
                        {
                            "head": text,
                            "hier": [head for _, head in hier],
                            "lines": [text],
                        }
                    )
                else:
                    if clauses:
                        clauses[-1]["lines"].append(text)
                continue

        return self._build_nodes(clauses + tables, doc, doc_code, model_pipeline)

    def _build_nodes(
        self,
        clauses: List[Dict[str, Any]],
        doc: BaseNode,
        doc_code: str,
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
                doc_code,
                model_pipeline,
                self.user_prompt,
                self.system_prompt,
            )
            
            metadata = {
                "topic": extracted["topic"],
                "content": content,
                "hier": clause["hier"],
                "cites": extracted["cites"],
            }

            print(f"{content} \n - topic: {metadata['topic']} \n - hierarchy: {metadata['hier']} \n - cites: {metadata['cites']}")

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

        # Only topic should contribute to the embedding
        node.excluded_embed_metadata_keys.extend(
            [
                "content",
                "hier",
                "cites",
            ]
        )
        
        node.excluded_llm_metadata_keys.extend(
            ["cites"]
        )
        
        return node
