"""Retrieval + Claude generation, glued together with LlamaIndex.

We don't use the off-the-shelf `RetrieverQueryEngine` because we want full
control over the prompt — specifically the [#1], [#2] citation style that
the API contract and Streamlit UI both depend on. The retriever and the LLM
both come from LlamaIndex; only the prompt assembly is ours.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.anthropic import Anthropic

from src.config import get_settings
from src.retrieval.embeddings import get_embed_model
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
def get_llm() -> Anthropic:
    s = get_settings()
    return Anthropic(model=s.anthropic_model, api_key=s.anthropic_api_key, max_tokens=1024)


def _retriever(top_k: int, source_filter: str | None):
    index = VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )
    filters = None
    if source_filter:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=source_filter)]
        )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


def retrieve(question: str, top_k: int, source_filter: str | None = None) -> list[NodeWithScore]:
    return _retriever(top_k, source_filter).retrieve(question)


def _build_context(nodes: list[NodeWithScore]) -> str:
    lines = []
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        src = meta.get("file_name", "unknown")
        page = meta.get("page_label", "?")
        lines.append(f"[#{i}] (source: {src}, page: {page})\n{n.node.get_content()}")
    return "\n\n---\n\n".join(lines)


def _sources_payload(nodes: list[NodeWithScore]) -> list[dict]:
    out = []
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        out.append(
            {
                "label": f"#{i}",
                "source": meta.get("file_name"),
                "page": meta.get("page_label"),
                "score": round(n.score or 0.0, 4),
                "preview": n.node.get_content()[:200],
            }
        )
    return out


def _messages(question: str, nodes: list[NodeWithScore]) -> list[ChatMessage]:
    user = f"Context:\n{_build_context(nodes)}\n\nQuestion: {question}"
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user),
    ]


def answer(
    question: str, top_k: int, source_filter: str | None = None
) -> tuple[GenerationResult, list[NodeWithScore]]:
    """Retrieve + generate. Returns the answer payload AND the raw nodes
    so callers can detect "no hits" without re-retrieving."""
    nodes = retrieve(question, top_k, source_filter)
    if not nodes:
        return GenerationResult(answer="", sources=[]), nodes

    response = get_llm().chat(_messages(question, nodes))
    return (
        GenerationResult(answer=response.message.content or "", sources=_sources_payload(nodes)),
        nodes,
    )


def stream_answer(
    question: str, top_k: int, source_filter: str | None = None
) -> tuple[Iterator[str], list[dict]]:
    """Retrieve + stream. Returns (token_iterator, sources_payload).
    Sources are returned eagerly so SSE callers can emit them before the first token."""
    nodes = retrieve(question, top_k, source_filter)
    if not nodes:
        return iter(()), []

    stream = get_llm().stream_chat(_messages(question, nodes))

    def deltas() -> Iterator[str]:
        for chunk in stream:
            if chunk.delta:
                yield chunk.delta

    return deltas(), _sources_payload(nodes)
