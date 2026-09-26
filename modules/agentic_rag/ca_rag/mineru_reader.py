"""Read a corpus of regulations into ``Document`` objects with MinerU.

The reader plays the role ``SimpleDirectoryReader`` plays for plain text, but
the text of a regulation is not enough: the clause chunker downstream needs to
know the role every block plays on the page. So each document is laid out by
MinerU and the resulting blocks are carried in its metadata.

    reader = MineruDirectoryReader(
        input_dir="data/RAG/test",
        corpus_csv="configs/corpus.csv",
        cache_dir="outputs/carag_cache",
    )
    documents = reader.load_data()

Given a ``cache_dir`` the layout analysis is cached per document, so only the
first run pays for it; without one every run lays the corpus out again.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_EXTS = {".pdf", ".txt", ".md", ".docx"}


class MineruDirectoryReader(BaseReader):
    def __init__(
        self,
        input_dir: str,
        corpus_csv: Optional[str] = None,
        cache_dir: Optional[str] = None,
        tier: str = "standard",
        force_reparse: bool = False,
        required_exts: Optional[Set[str]] = None,
        recursive: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.corpus_csv = Path(corpus_csv) if corpus_csv else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tier = tier
        self.force_reparse = force_reparse
        self.required_exts = required_exts or DEFAULT_CORPUS_EXTS
        self.recursive = recursive

        if not self.input_dir.is_dir():
            raise ValueError(f"No corpus directory at {input_dir}")

    def _input_files(self) -> List[Path]:
        """Collect the corpus files, in a stable order."""
        walk = self.input_dir.rglob if self.recursive else self.input_dir.glob
        paths = sorted(
            p for p in walk("*")
            if p.is_file() and p.suffix.lower() in self.required_exts
        )
        if not paths:
            raise ValueError(f"No corpus files found in {self.input_dir}")
        return paths

    def _load_corpus_config(self) -> Dict[str, Dict[str, str]]:
        """
        Read the per-document settings, keyed by document name.

        Front matter is declared rather than detected: a cover, a foreword and a
        table of contents repeat the headings of the body and would be chunked as
        clauses of their own, and no layout model can tell that they are not
        wanted.
        """
        if not self.corpus_csv or not self.corpus_csv.exists():
            logger.warning(f"No corpus config at {self.corpus_csv}, parsing every document in full")
            return {}

        with open(self.corpus_csv, newline="", encoding="utf-8") as f:
            config = {
                row["name"].strip(): {
                    "code": (row.get("code") or "").strip(),
                    "skip": (row.get("skip") or "").strip(),
                }
                for row in csv.DictReader(f)
                if (row.get("name") or "").strip()
            }
        logger.info(f"Loaded settings for {len(config)} documents from {self.corpus_csv}")
        return config

    def _page_range_of(self, path: Path, skip: str) -> str:
        """
        Translate the pages to skip into the page range MinerU keeps.

        MinerU counts pages from 1 and, given no range at all, reads only the
        first ten, so the full range has to be spelled out even when nothing is
        skipped.
        """
        from pypdf import PdfReader
        from mineru.parser.page_range import format_page_range, parse_page_range_set

        # Calculate the number of pages in the document.
        n_pages = len(PdfReader(str(path)).pages)

        # Translate the pages to skip into the page range MinerU keeps.
        skipped = parse_page_range_set(skip) if skip else set()

        # Check that the resulting page range is not empty.
        keep = [p for p in range(1, n_pages + 1) if p not in skipped]
        if not keep:
            raise ValueError(f"{path.name}: skip {skip!r} would skip all {n_pages} pages")

        return format_page_range(keep)

    def _parse_layout(self, path: Path, page_range: str) -> List[Dict]:
        """
        Lay out one document with MinerU, caching the result.

        The analysis runs a vision model over every page, which dominates the
        build time, while the chunking downstream of it is cheap and gets
        iterated on. The two are separated by this cache. Without a
        ``cache_dir`` the document is still laid out, just never cached.
        """
        cache_path = self.cache_dir / f"{path.stem}.json" if self.cache_dir else None
        if cache_path and cache_path.exists() and not self.force_reparse:
            from mineru.parser.base import ParseResult

            result = ParseResult.from_json(cache_path.read_text(encoding="utf-8"))
            logger.info(f"{path.name}: layout read from {cache_path.name}")
        else:
            from mineru.parser.mineru_parser import MinerUParser

            logger.info(f"{path.name}: running MinerU over pages {page_range}")
            result = MinerUParser(tier=self.tier).parse(str(path), page_range=page_range)
            if cache_path:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(result.to_json(), encoding="utf-8")
                logger.info(f"{path.name}: layout cached to {cache_path.name}")

        return result.structured_content().get("pages", [])

    def lazy_load_data(self) -> Iterator[Document]:
        """
        Yield one ``Document`` per corpus file as its layout finishes.

        The MinerU blocks are carried on the document so that
        ``ClauseNodeParser`` can chunk on typed blocks instead of having to
        recover the role of each line from its shape.
        """
        config = self._load_corpus_config()
        for path in self._input_files():
            settings = config.get(path.stem, {})
            if path.stem not in config:
                logger.warning(f"{path.name}: absent from the corpus config, parsing in full")

            # Translate the pages to skip into the page range MinerU keeps.
            page_range = self._page_range_of(path, settings.get("skip", ""))

            # Parse the document with MinerU, caching the result if requested.
            pages = self._parse_layout(path, page_range)

            # Yield a document with the MinerU blocks
            logger.info(f"{path.name}: {len(pages)} pages laid out")
            yield Document(
                # TODO: 是否返回正常的正文在 text 里比较好
                text=f"{path.name} ({len(pages)} pages)",
                metadata={
                    "file_name": path.name,
                    "file_path": str(path),
                    "code": settings.get("code", ""),
                    "mineru_pages": pages,
                },
                # bookkeeping for the parser, never embedded or shown to an LLM
                excluded_embed_metadata_keys=["mineru_pages", "file_path"],
                excluded_llm_metadata_keys=["mineru_pages", "file_path"],
            )
