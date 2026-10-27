from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from llama_index.core.evaluation.retrieval.metrics import NDCG, HitRate, Recall

_WORD_RE = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int = 3) -> Counter:
    """Word n-grams of a passage, lower-cased with punctuation and markup dropped."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return Counter([tuple(words)]) if words else Counter()
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def overlap(gold: str, chunk: str, n: int = 3) -> float:
    """Overlap coefficient of the word n-grams of two passages.

    Normalised by the shorter passage, so a chunk lying inside a long clause
    and a chunk that swallows a short clause both count as covering it.
    """
    g, c = ngrams(gold, n), ngrams(chunk, n)
    if not g or not c:
        return 0.0
    return sum((g & c).values()) / min(sum(g.values()), sum(c.values()))


def label_hits(retrieved: Sequence[str], gold: Sequence[str], threshold: float) -> List[str]:
    """Relevance ids for a ranked list, independent of how the corpus was chunked.

    Each retrieved chunk is labelled with the gold passage it covers best
    (``gold-j``), or a unique ``miss-i``. A gold passage is claimed by its first
    covering chunk only, so a clause split over two chunks is not counted twice.
    """
    labels: List[str] = []
    claimed: set[int] = set()
    for i, chunk in enumerate(retrieved):
        score, best = max(
            ((overlap(passage, chunk), j) for j, passage in enumerate(gold) if j not in claimed),
            default=(0.0, -1),
        )
        if score >= threshold:
            claimed.add(best)
            labels.append(f"gold-{best}")
        else:
            labels.append(f"miss-{i}")
    return labels


class RetrievalMetrics:
    """Compute and track retrieval metrics using llama_index.

    Recall@k, NDCG@k and Hit Rate@1, averaged over queries. Systems chunk the
    corpus differently, so relevance is judged by text: a retrieved passage is
    relevant when its word 3-grams overlap a gold passage by ``threshold``.
    """

    def __init__(self, ks: Sequence[int] = (5, 10), threshold: float = 0.5):
        """
        Args:
            ks: Cut-offs of Recall and NDCG.
            threshold: Overlap coefficient a retrieved passage needs with a gold passage to be relevant.
        """
        self.threshold = threshold

        # metric instances, each with the depth of the ranking it scores
        self._metrics = {
            **{f"Recall@{k}": (Recall(), k) for k in ks},
            **{f"NDCG@{k}": (NDCG(), k) for k in ks},
            "HitRate@1": (HitRate(), 1),
        }
        self._scores: List[Dict[str, float]] = []

    def update(self, pred: Sequence[str], target: Sequence[str]) -> Dict[str, float]:
        """
        Score and accumulate one query.

        Args:
            pred: Texts of the retrieved passages, ranked best first.
            target: Texts of the gold passages, unordered.

        Returns:
            dict of metric-name -> score of this query.
        """
        if not target:
            raise ValueError("A query needs at least one gold passage")

        labels = label_hits(pred, target, self.threshold)
        expected = [f"gold-{j}" for j in range(len(target))]

        scores = {}
        for name, (metric, k) in self._metrics.items():
            # llama_index rejects an empty ranking; nothing retrieved scores zero
            scores[name] = (
                metric.compute(expected_ids=expected, retrieved_ids=labels[:k]).score
                if labels[:k] else 0.0
            )
        self._scores.append(scores)
        return scores

    def compute(self) -> Dict[str, float]:
        """Compute final metrics across all accumulated queries.

        Returns a dict of metric-name -> mean score.
        """
        if not self._scores:
            return {name: float("nan") for name in self._metrics}

        return {
            name: math.fsum(s[name] for s in self._scores) / len(self._scores)
            for name in self._metrics
        }

    def reset(self):
        """Reset all metrics for new evaluation."""
        self._scores.clear()

    def summarize(self):
        """
        Summarize all computed metrics. Each column header is suffixed with an
        arrow marking that every retrieval metric is better high (↑).
        """
        metrics = self.compute()

        title = ""
        value = ""
        for metric_name, metric_value in metrics.items():
            header = f"{metric_name}↑"
            title += f"{header:>12s}"
            value += f"{metric_value:>12.4f}"

        return title + "\n" + value
