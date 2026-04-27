"""
LlamaIndex generation walkthrough — close the RAG loop with query engines.

The pieces:
    - VectorStoreIndex      — already familiar from earlier walkthroughs
    - Anthropic LLM         — LlamaIndex's wrapper around the Claude API
    - RetrieverQueryEngine  — glues retriever + LLM + a response synthesizer
    - .query() / .stream()  — the call you actually make

We'll:
    1. Build a small RAG corpus (Document → SentenceSplitter → embed → index)
    2. Ask Claude WITHOUT context (baseline)
    3. Ask Claude through a RetrieverQueryEngine (the RAG payoff)
    4. The honesty test — out-of-scope question
    5. Stream a response token-by-token

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY and ANTHROPIC_API_KEY in .env

Run:
    uv run python -m scripts.learn.generation_walkthrough
"""
from __future__ import annotations

import textwrap

from llama_index.core import (
    Document,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rich import print
from rich.panel import Panel
from rich.rule import Rule

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "generation_walkthrough"
settings = get_settings()

embed_model = VoyageEmbedding(
    model_name=settings.voyage_model,
    voyage_api_key=settings.voyage_api_key,
)
llm = Anthropic(
    model=settings.anthropic_model,
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
)
qdrant = QdrantClient(url=settings.qdrant_url)


DOCUMENT = textwrap.dedent("""\
    # Acme Corp Remote Work Policy

    ## Eligibility
    All full-time employees who have completed their 90-day onboarding period are eligible
    to apply for remote work. This includes engineering, design, product, marketing, and
    operations roles. Some positions, particularly those requiring physical presence such
    as facilities and on-site IT support, are not eligible.

    ## Application Process
    To apply for remote work, employees must submit a Remote Work Request form through
    Workday at least 30 days before the intended start date. The form requires manager
    approval and a brief justification. Decisions are typically returned within 10 business days.

    ## Equipment and Stipends
    Approved remote workers receive a one-time equipment stipend of $1,500 to set up their
    home office. Acme additionally provides a monthly internet reimbursement of $75. Employees
    are responsible for ensuring their home network meets the minimum bandwidth requirement
    of 50 Mbps download and 10 Mbps upload.

    ## Security Requirements
    Remote workers must use the company-issued laptop for all work activities. Personal
    devices may not access internal systems. All remote workers must enable full-disk
    encryption, use the company VPN when accessing internal resources, and complete the
    quarterly security training. Failure to comply may result in revocation of remote work privileges.

    ## Hybrid Schedules
    Employees may opt for a hybrid schedule, defined as 2-3 days per week in the office.
    Hybrid employees do not receive the equipment stipend but do receive the internet reimbursement.
""")


# ---------------------------------------------------------------------------
# 0. Build the corpus
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]0. Build the corpus (Document → split → embed → index)"))

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)

vector_store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
nodes = splitter.get_nodes_from_documents([Document(text=DOCUMENT, metadata={"source": "remote_work_policy.md"})])
print(f"Split into {len(nodes)} nodes.")

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=embed_model,
)
print("Indexed.\n")


# ---------------------------------------------------------------------------
# 1. Naked Claude (no context)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Ask Claude WITHOUT any context"))

question = "What is the equipment stipend for remote workers at Acme Corp?"
print(f"[bold]Question:[/bold] {question}\n")

naked = llm.complete(question)
print(Panel(naked.text, title="Naked Claude", border_style="yellow"))
print("[dim]Claude has no idea about Acme Corp's specific policies.[/dim]\n")


# ---------------------------------------------------------------------------
# 2. Grounded via a RetrieverQueryEngine
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Ask Claude through a RetrieverQueryEngine (RAG)"))

# Customize the prompt so Claude cites passages with [#1], [#2] labels —
# the same convention our production app uses.
GROUNDED_TEMPLATE = PromptTemplate(
    "You are a helpful research assistant. Answer the user's question using ONLY the context "
    "passages below. Each passage is labeled like [#1], [#2], etc. Cite the passages you used "
    "inline using the same labels. If the context does not contain the answer, say so honestly.\n"
    "\nContext:\n{context_str}\n"
    "\nQuestion: {query_str}\n"
    "Answer: "
)

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode=ResponseMode.COMPACT,
    text_qa_template=GROUNDED_TEMPLATE,
)

response = query_engine.query(question)
print(Panel(str(response), title="Grounded Claude", border_style="green"))

print("\n[bold]Source nodes used:[/bold]")
for i, sn in enumerate(response.source_nodes, start=1):
    print(f"  [#{i}]  score={sn.score:.4f}  "
          f"source={sn.node.metadata.get('source')}  "
          f"text={sn.node.get_content()[:100]!r}...")


# ---------------------------------------------------------------------------
# 3. The honesty test
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. The honesty test — out-of-scope question"))

oos_question = "What's Acme Corp's parental leave policy?"
print(f"[bold]Question:[/bold] {oos_question}")
print("[dim](This document doesn't say anything about parental leave.)[/dim]\n")

oos_response = query_engine.query(oos_question)
print(Panel(str(oos_response), title="Grounded Claude (out-of-scope)", border_style="red"))
print("[dim]A well-grounded RAG should say 'I don't know' rather than confabulate.[/dim]\n")


# ---------------------------------------------------------------------------
# 4. Streaming
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Streaming response (watch tokens arrive live)"))

stream_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode=ResponseMode.COMPACT,
    text_qa_template=GROUNDED_TEMPLATE,
    streaming=True,
)
stream_question = "What's the application process for remote work?"
print(f"[bold]Question:[/bold] {stream_question}\n")
print("[bold]Streamed answer:[/bold]")

streaming_response = stream_engine.query(stream_question)
streaming_response.print_response_stream()
print("\n")
print("[dim]Same answer as the non-streaming engine, just delivered token by token. "
      "Plug this into FastAPI's StreamingResponse for an SSE endpoint.[/dim]\n")


# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Done"))
print(f"Collection '{COLLECTION}' preserved at http://localhost:6333/dashboard")
print("[bold green]✓ Walkthrough complete. End-to-end RAG via LlamaIndex.[/bold green]")
