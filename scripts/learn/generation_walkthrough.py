"""
Claude generation walkthrough — close the RAG loop.

This is the payoff. We're going to:

    1. Build a small RAG corpus (using everything you've learned: chunk + embed + index)
    2. Make a NAKED call to Claude (no context) — see how it does
    3. Make a GROUNDED call (with retrieved context) — see the difference
    4. Watch what happens when the context doesn't actually contain the answer
    5. Compare temperature 0.0 vs 0.7 — see grounding vs creativity tradeoff
    6. Stream a response token-by-token — see the UX upgrade

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY and ANTHROPIC_API_KEY in environment

Run:
    uv run python -m scripts.learn.generation_walkthrough
"""
from __future__ import annotations

import os
import textwrap

import voyageai
from anthropic import Anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich import print
from rich.panel import Panel
from rich.rule import Rule

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "generation_walkthrough"
EMBED_MODEL = "voyage-3"
DIM = 1024
CLAUDE_MODEL = "claude-sonnet-4-6"

if not os.getenv("VOYAGE_API_KEY"):
    raise SystemExit("Set VOYAGE_API_KEY first.")
if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY first.")

vo = voyageai.Client()
qdrant = QdrantClient(url="http://localhost:6333")
claude = Anthropic()


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
# 0. Setup the corpus
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]0. Build the corpus (chunk → embed → index)"))

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_text(DOCUMENT)
print(f"Chunked into {len(chunks)} pieces.")

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

embeddings = vo.embed(chunks, model=EMBED_MODEL, input_type="document").embeddings
qdrant.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(
            id=i,
            vector=v,
            payload={"text": t, "source": "remote_work_policy.md", "chunk_index": i},
        )
        for i, (t, v) in enumerate(zip(chunks, embeddings))
    ],
)
print(f"Indexed {len(chunks)} chunks. Ready.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retrieve(question: str, k: int = 3) -> list[dict]:
    qvec = vo.embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION, query=qvec, limit=k, with_payload=True
    ).points
    return [
        {
            "label": f"#{i + 1}",
            "text": h.payload["text"],
            "source": h.payload["source"],
            "chunk_index": h.payload["chunk_index"],
            "score": h.score,
        }
        for i, h in enumerate(hits)
    ]


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h['label']}] (source: {h['source']}, chunk: {h['chunk_index']})\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_GROUNDED = """You are a helpful research assistant.
Answer the user's question using ONLY the provided context passages.
Each passage is labeled like [#1], [#2], etc.
Cite the passages you used inline using the same labels (e.g. "as shown in [#2]").
If the context does not contain the answer, say so honestly. Do not invent facts."""

SYSTEM_NAKED = "You are a helpful assistant."


def ask_claude(system: str, user_msg: str, temperature: float = 0.0) -> str:
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# 1. NAKED Claude — no context at all
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Ask Claude WITHOUT any context (the baseline)"))

question = "What is the equipment stipend for remote workers at Acme Corp?"
print(f"[bold]Question:[/bold] {question}\n")

naked_answer = ask_claude(SYSTEM_NAKED, question)
print(Panel(naked_answer, title="Naked Claude", border_style="yellow"))
print(
    "[dim]Notice: Claude has NO IDEA about Acme Corp's specific policies. It either makes "
    "something up, gives generic info, or admits ignorance. This is what 'no RAG' looks like.[/dim]\n"
)


# ---------------------------------------------------------------------------
# 2. GROUNDED Claude — with retrieved context
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Ask Claude WITH retrieved context (the RAG payoff)"))

hits = retrieve(question, k=3)
print(f"Retrieved {len(hits)} chunks. Top scores: {[round(h['score'], 3) for h in hits]}\n")

context = format_context(hits)
user_msg = f"Context:\n{context}\n\nQuestion: {question}"

print("[dim]The actual prompt being sent to Claude:[/dim]")
print(Panel(user_msg, title="User message", border_style="blue", width=100))
print()

grounded_answer = ask_claude(SYSTEM_GROUNDED, user_msg)
print(Panel(grounded_answer, title="Grounded Claude", border_style="green"))
print(
    "[dim]Notice: Claude answers with the SPECIFIC fact from the document AND cites it. "
    "This is the entire promise of RAG — turning a generic LLM into a domain expert.[/dim]\n"
)


# ---------------------------------------------------------------------------
# 3. The honesty test
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. The honesty test — out-of-scope question"))

oos_question = "What's Acme Corp's parental leave policy?"
print(f"[bold]Question:[/bold] {oos_question}")
print("[dim](This document doesn't say anything about parental leave.)[/dim]\n")

oos_hits = retrieve(oos_question, k=3)
print(f"Retrieved chunks anyway (top scores: {[round(h['score'], 3) for h in oos_hits]})")
print("[dim]Note the lower scores — vector search returns SOMETHING, even when nothing fits.[/dim]\n")

oos_context = format_context(oos_hits)
oos_user = f"Context:\n{oos_context}\n\nQuestion: {oos_question}"
oos_answer = ask_claude(SYSTEM_GROUNDED, oos_user)
print(Panel(oos_answer, title="Grounded Claude (out-of-scope)", border_style="red"))
print(
    "[dim]A well-grounded RAG should say 'I don't know' rather than confabulate. "
    "If you see Claude make something up here, the system prompt isn't strict enough.[/dim]\n"
)


# ---------------------------------------------------------------------------
# 4. Temperature comparison
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Temperature: 0.0 vs 0.7"))

temp_question = "Summarize the security requirements for remote workers."
temp_hits = retrieve(temp_question, k=3)
temp_user = f"Context:\n{format_context(temp_hits)}\n\nQuestion: {temp_question}"

print(f"[bold]Question:[/bold] {temp_question}\n")

answer_strict = ask_claude(SYSTEM_GROUNDED, temp_user, temperature=0.0)
answer_loose = ask_claude(SYSTEM_GROUNDED, temp_user, temperature=0.7)

print(Panel(answer_strict, title="temperature=0.0 (deterministic)", border_style="cyan"))
print(Panel(answer_loose, title="temperature=0.7 (creative)", border_style="magenta"))
print(
    "[dim]temp=0.0 is reproducible and tightly grounded — same input, same output. "
    "temp=0.7 has more variation and may rephrase more creatively. For RAG, lower is "
    "usually better.[/dim]\n"
)


# ---------------------------------------------------------------------------
# 5. Streaming
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Streaming response (watch tokens arrive live)"))

stream_question = "What's the application process for remote work?"
stream_hits = retrieve(stream_question, k=3)
stream_user = f"Context:\n{format_context(stream_hits)}\n\nQuestion: {stream_question}"

print(f"[bold]Question:[/bold] {stream_question}\n")
print("[bold]Streamed answer:[/bold]")

with claude.messages.stream(
    model=CLAUDE_MODEL,
    max_tokens=1024,
    temperature=0.0,
    system=SYSTEM_GROUNDED,
    messages=[{"role": "user", "content": stream_user}],
) as stream:
    for delta in stream.text_stream:
        print(delta, end="", flush=True)
print("\n")

print(
    "[dim]Same answer as non-streaming, but tokens arrive incrementally. This is what "
    "you'll wire into the FastAPI endpoint via Server-Sent Events for the UI.[/dim]\n"
)


# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Done"))
print(f"Collection '{COLLECTION}' preserved at http://localhost:6333/dashboard")
print("[bold green]✓ Walkthrough complete. You've seen end-to-end RAG.[/bold green]")
