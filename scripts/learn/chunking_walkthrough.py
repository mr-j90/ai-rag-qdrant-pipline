"""
LangChain text splitters walkthrough — side by side.

LangChain's text splitters live in `langchain-text-splitters`. The relevant
ones for prose:

    - CharacterTextSplitter             — splits on a single separator
    - RecursiveCharacterTextSplitter    — tries paragraph → line → sentence → word
                                          (this is what our project uses; the
                                          recommended general-purpose default)
    - TokenTextSplitter                 — fixed token windows via tiktoken

Each splitter has two methods:
    - .split_text(str) -> list[str]
    - .split_documents([Document]) -> list[Document]   ← preserves metadata

This walkthrough takes one document, runs four variants, indexes each into
its own Qdrant collection, and runs the same query against each.

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY in .env

Run:
    uv run python -m scripts.learn.chunking_walkthrough
"""
from __future__ import annotations

import textwrap

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print
from rich.rule import Rule
from rich.table import Table

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
settings = get_settings()
embeddings = VoyageAIEmbeddings(
    model=settings.voyage_model,
    api_key=settings.voyage_api_key,
)
qdrant = QdrantClient(url=settings.qdrant_url)


DOCUMENT = textwrap.dedent("""\
    # Acme Corp Remote Work Policy

    ## Eligibility

    All full-time employees who have completed their 90-day onboarding period are eligible
    to apply for remote work. This includes engineering, design, product, marketing, and
    operations roles. Some positions, particularly those requiring physical presence such
    as facilities and on-site IT support, are not eligible for remote work arrangements.

    ## Application Process

    To apply for remote work, employees must submit a Remote Work Request form through
    Workday at least 30 days before the intended start date. The form requires manager
    approval and a brief justification describing how the employee will maintain
    productivity and collaboration in a remote setting. Decisions are typically returned
    within 10 business days.

    ## Equipment and Stipends

    Approved remote workers receive a one-time equipment stipend of $1,500 to set up their
    home office. This stipend covers monitors, chairs, desks, and similar equipment. Acme
    additionally provides a monthly internet reimbursement of $75 for primary remote
    workers. Employees are responsible for ensuring their home network meets the minimum
    bandwidth requirement of 50 Mbps download and 10 Mbps upload.

    ## Security Requirements

    Remote workers must use the company-issued laptop for all work activities. Personal
    devices may not access internal systems. All remote workers must enable full-disk
    encryption, use the company VPN when accessing internal resources, and complete the
    quarterly security training. Failure to comply with security requirements may result
    in revocation of remote work privileges.

    ## Hybrid Schedules

    Employees may opt for a hybrid schedule, defined as 2-3 days per week in the office.
    Hybrid employees do not receive the equipment stipend but do receive the internet
    reimbursement. Hybrid schedules require the same application process as fully remote
    arrangements.
""")

source_doc = Document(page_content=DOCUMENT, metadata={"source": "remote_work_policy.md"})


# ---------------------------------------------------------------------------
# 1. CharacterTextSplitter — one separator only
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. CharacterTextSplitter (single separator, no overlap)"))

char_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=400, chunk_overlap=0
)
char_chunks = char_splitter.split_documents([source_doc])
print(f"Got {len(char_chunks)} chunks. Sizes: {[len(c.page_content) for c in char_chunks]}\n")
for i, c in enumerate(char_chunks):
    text = c.page_content
    print(f"[bold]Chunk {i}[/bold] ({len(text)} chars):")
    print(f"  start: {text[:60]!r}...")
    print(f"  end:   ...{text[-60:]!r}\n")
print("[dim]One separator means anything bigger than chunk_size that doesn't contain "
      "the separator stays in one chunk — common gotcha.[/dim]")


# ---------------------------------------------------------------------------
# 2. RecursiveCharacterTextSplitter, no overlap
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. RecursiveCharacterTextSplitter, no overlap"))

recursive_no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""],
)
no_overlap_chunks = recursive_no_overlap.split_documents([source_doc])
print(f"Got {len(no_overlap_chunks)} chunks. Sizes: {[len(c.page_content) for c in no_overlap_chunks]}\n")
for i, c in enumerate(no_overlap_chunks):
    text = c.page_content
    print(f"[bold]Chunk {i}[/bold] ({len(text)} chars):")
    print(f"  start: {text[:60]!r}...")
    print(f"  end:   ...{text[-60:]!r}\n")
print("[dim]Better — falls back through paragraph → line → sentence → word boundaries.[/dim]")


# ---------------------------------------------------------------------------
# 3. RecursiveCharacterTextSplitter WITH overlap (the production default)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. RecursiveCharacterTextSplitter WITH overlap (production default)"))

recursive_overlap = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
overlap_chunks = recursive_overlap.split_documents([source_doc])
print(f"Got {len(overlap_chunks)} chunks.\n")
for i in range(len(overlap_chunks) - 1):
    end = overlap_chunks[i].page_content[-60:]
    start = overlap_chunks[i + 1].page_content[:60]
    print(f"[bold]Chunk {i} → {i + 1} boundary[/bold]")
    print(f"  end of {i}:    ...{end!r}")
    print(f"  start of {i + 1}:  {start!r}...\n")
print("[dim]Deliberate duplication at boundaries: a fact near the seam appears in both "
      "chunks, so retrieval can't lose it.[/dim]")


# ---------------------------------------------------------------------------
# 4. TokenTextSplitter — counts tokens, not characters
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. TokenTextSplitter (tiktoken-based)"))

token_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
token_chunks = token_splitter.split_documents([source_doc])
print(f"Got {len(token_chunks)} chunks (sized by tokens, not chars).\n")
for i, c in enumerate(token_chunks):
    text = c.page_content
    snippet = text[:120].replace("\n", " ")
    print(f"[bold]Chunk {i}[/bold] ({len(text)} chars): {snippet}...")
print("\n[dim]Useful when you care about hitting an LLM's token budget exactly. "
      "Tradeoff: chunks ignore document structure entirely.[/dim]")


# ---------------------------------------------------------------------------
# 5. Index each variant and compare retrieval
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Compare retrieval across strategies"))


def index_chunks(name: str, chunks: list[Document]) -> QdrantVectorStore:
    coll = f"chunking_{name}"
    if coll in {c.name for c in qdrant.get_collections().collections}:
        qdrant.delete_collection(coll)
    qdrant.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    vs = QdrantVectorStore(client=qdrant, collection_name=coll, embedding=embeddings)
    vs.add_documents(chunks)
    return vs


print("Indexing all four variants...")
vs_char = index_chunks("character", char_chunks)
vs_no_overlap = index_chunks("recursive_no_overlap", no_overlap_chunks)
vs_overlap = index_chunks("recursive_with_overlap", overlap_chunks)
vs_token = index_chunks("token", token_chunks)
print("  ✓ Indexed.\n")

queries = [
    "How much is the equipment stipend for remote workers?",
    "What internet speed do I need at home?",
    "Can I use my personal laptop for work?",
    "Who is eligible for remote work?",
]

for q in queries:
    print(f"[bold]Q:[/bold] {q}\n")
    table = Table(show_lines=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Top retrieved chunk (truncated)", overflow="fold", max_width=70)

    for label, vs in [
        ("Character", vs_char),
        ("Recursive (no overlap)", vs_no_overlap),
        ("Recursive (with overlap)", vs_overlap),
        ("Token", vs_token),
    ]:
        hits = vs.similarity_search_with_score(q, k=1)
        if not hits:
            table.add_row(label, "-", "(no hits)")
            continue
        doc, score = hits[0]
        snippet = doc.page_content[:200].replace("\n", " ")
        if len(doc.page_content) > 200:
            snippet += "..."
        table.add_row(label, f"{score:.4f}", snippet)
    print(table)
    print()


# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Done"))
print("Collections preserved:")
print("  - chunking_character")
print("  - chunking_recursive_no_overlap")
print("  - chunking_recursive_with_overlap")
print("  - chunking_token")
print("Compare them at http://localhost:6333/dashboard.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
