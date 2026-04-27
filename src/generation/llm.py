"""Retrieval + Claude generation, glued together with LangChain (LCEL).

We don't use the off-the-shelf `RetrievalQA` chain because we want full
control over the prompt — specifically the [#1], [#2] citation style that
the API contract and Streamlit UI both depend on. The retriever and the LLM
both come from LangChain; only the prompt assembly is ours.

The "answer" half is LCEL — `prompt | llm | parser`. The "retrieval" half is
imperative because we need the retrieved docs both for the prompt AND for the
sources payload that goes back to the UI.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import get_settings
from src.retrieval.store import get_vector_store


SYSTEM_PROMPT = """You are a helpful research assistant.
Answer the user's question using ONLY the provided context passages.
Each passage is labeled like [#1], [#2], etc.
Cite the passages you used inline using the same labels (e.g. "... as shown in [#2]").
If the context does not contain the answer, say so honestly. Do not invent facts."""


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]


@lru_cache
def get_llm() -> ChatAnthropic:
    s = get_settings()
    return ChatAnthropic(
        model=s.anthropic_model,
        api_key=s.anthropic_api_key,
        max_tokens=1024,
    )


@lru_cache
def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )


@lru_cache
def get_chain():
    """LCEL chain: prompt | llm | StrOutputParser. Same chain for sync + stream."""
    return get_prompt() | get_llm() | StrOutputParser()


def _retrieve(question: str, top_k: int, source_filter: str | None) -> list[Document]:
    """Use similarity_search_with_score so we can keep scores in the sources payload.
    The standard `as_retriever()` path drops them."""
    qfilter = None
    if source_filter:
        # LangChain's QdrantVectorStore stores metadata nested under the `metadata`
        # payload key, so the filter path is `metadata.source`.
        qfilter = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_filter))]
        )
    pairs = get_vector_store().similarity_search_with_score(
        question, k=top_k, filter=qfilter
    )
    docs: list[Document] = []
    for doc, score in pairs:
        doc.metadata["_score"] = float(score)
        docs.append(doc)
    return docs


def _build_context(docs: list[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        lines.append(f"[#{i}] (source: {src}, page: {page})\n{d.page_content}")
    return "\n\n---\n\n".join(lines)


def _sources_payload(docs: list[Document]) -> list[dict]:
    out = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        out.append(
            {
                "label": f"#{i}",
                "source": meta.get("source"),
                "page": meta.get("page"),
                "score": round(meta["_score"], 4) if "_score" in meta else None,
                "preview": d.page_content[:200],
            }
        )
    return out


def answer(
    question: str, top_k: int, source_filter: str | None = None
) -> tuple[GenerationResult, list[Document]]:
    """Retrieve + generate. Returns the answer payload AND the raw docs
    so callers can detect "no hits" without re-retrieving."""
    docs = _retrieve(question, top_k, source_filter)
    if not docs:
        return GenerationResult(answer="", sources=[]), docs

    text = get_chain().invoke({"context": _build_context(docs), "question": question})
    return GenerationResult(answer=text, sources=_sources_payload(docs)), docs


def stream_answer(
    question: str, top_k: int, source_filter: str | None = None
) -> tuple[Iterator[str], list[dict]]:
    """Retrieve + stream. Returns (token_iterator, sources_payload).
    Sources are returned eagerly so SSE callers can emit them before the first token."""
    docs = _retrieve(question, top_k, source_filter)
    if not docs:
        return iter(()), []

    chunks = get_chain().stream({"context": _build_context(docs), "question": question})
    return chunks, _sources_payload(docs)
