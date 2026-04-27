"""Claude wrapper for the generation step. Builds a context-grounded prompt
from retrieved chunks and asks Claude to answer with inline citations."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from anthropic import Anthropic

from src.config import get_settings
from src.retrieval.store import SearchResult


SYSTEM_PROMPT = """You are a helpful research assistant.
Answer the user's question using ONLY the provided context passages.
Each passage is labeled like [#1], [#2], etc.
Cite the passages you used inline using the same labels (e.g. "... as shown in [#2]").
If the context does not contain the answer, say so honestly. Do not invent facts."""


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]


def _build_context(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        src = r.metadata.get("source", "unknown")
        page = r.metadata.get("page", "?")
        lines.append(f"[#{i}] (source: {src}, page: {page})\n{r.text}")
    return "\n\n---\n\n".join(lines)


def _sources_payload(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "label": f"#{i}",
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "score": round(r.score, 4),
            "preview": r.text[:200],
        }
        for i, r in enumerate(results, start=1)
    ]


class ClaudeGenerator:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    def answer(self, question: str, results: list[SearchResult]) -> GenerationResult:
        context = _build_context(results)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return GenerationResult(answer=text, sources=_sources_payload(results))

    def stream_answer(self, question: str, results: list[SearchResult]) -> Iterator[str]:
        context = _build_context(results)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for delta in stream.text_stream:
                yield delta
