"""
Chunking walkthrough — see what good and bad chunking actually look like.

We'll take one document and chunk it three different ways:
    1. Naive fixed-size split (no overlap, ignores boundaries)
    2. RecursiveCharacterTextSplitter, no overlap
    3. RecursiveCharacterTextSplitter, with overlap (production pattern)

Then we'll embed all three versions, put them in three Qdrant collections,
and run the same query against each. You'll see retrieval quality change.

Prereqs:
    - docker compose up -d  (Qdrant running)
    - VOYAGE_API_KEY in environment

Run:
    uv run python -m scripts.learn.chunking_walkthrough
"""
from __future__ import annotations

import os
import textwrap

import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich import print
from rich.rule import Rule
from rich.table import Table

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
MODEL = "voyage-3"
DIM = 1024

api_key = os.getenv("VOYAGE_API_KEY")
if not api_key:
    raise SystemExit("Set VOYAGE_API_KEY first.")

vo = voyageai.Client(api_key=api_key)
qdrant = QdrantClient(url="http://localhost:6333")


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

# ---------------------------------------------------------------------------
# 1. Naive chunking — what NOT to do
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Naive fixed-size chunking (the baseline)"))


def naive_chunk(text: str, size: int = 400) -> list[str]:
    """Slice every `size` characters. No overlap, no boundary respect."""
    return [text[i : i + size] for i in range(0, len(text), size)]


naive_chunks = naive_chunk(DOCUMENT, size=400)
print(f"Got {len(naive_chunks)} chunks of ~400 chars each.\n")

for i, c in enumerate(naive_chunks):
    print(f"[bold]Chunk {i}[/bold] ({len(c)} chars):")
    print(f"  start: ...{c[:60]!r}")
    print(f"  end:   ...{c[-60:]!r}\n")

print(
    "[dim]Notice: chunks start and end mid-word, mid-sentence, sometimes mid-section-header. "
    "Embeddings of these will be lower quality because the model sees broken context.[/dim]"
)


# ---------------------------------------------------------------------------
# 2. Recursive splitter, NO overlap
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. RecursiveCharacterTextSplitter, no overlap"))

no_overlap = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = no_overlap.split_text(DOCUMENT)
print(f"Got {len(recursive_chunks)} chunks. Sizes: {[len(c) for c in recursive_chunks]}\n")

for i, c in enumerate(recursive_chunks):
    print(f"[bold]Chunk {i}[/bold] ({len(c)} chars):")
    print(f"  start: {c[:60]!r}...")
    print(f"  end:   ...{c[-60:]!r}\n")

print(
    "[dim]Better: chunks now break on paragraph and sentence boundaries. But the boundary "
    "between chunks is still a hard cut — info near the seam can be orphaned.[/dim]"
)


# ---------------------------------------------------------------------------
# 3. Recursive splitter, WITH overlap (the production pattern)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. RecursiveCharacterTextSplitter, with overlap"))

with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)
overlap_chunks = with_overlap.split_text(DOCUMENT)
print(f"Got {len(overlap_chunks)} chunks. Sizes: {[len(c) for c in overlap_chunks]}\n")

for i in range(len(overlap_chunks) - 1):
    end_of_current = overlap_chunks[i][-60:]
    start_of_next = overlap_chunks[i + 1][:60]
    print(f"[bold]Chunk {i} → Chunk {i + 1} boundary[/bold]")
    print(f"  end of {i}:    ...{end_of_current!r}")
    print(f"  start of {i + 1}:  {start_of_next!r}...")
    print()

print(
    "[dim]Now there's deliberate duplication at boundaries. A key sentence near a seam "
    "appears in BOTH chunks — so retrieval can't lose it.[/dim]"
)


# ---------------------------------------------------------------------------
# 4. Index all three versions and compare retrieval
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Compare retrieval quality across chunking strategies"))


def index_chunks(name: str, chunks: list[str]) -> str:
    """Create a fresh collection, embed chunks, upsert. Returns collection name."""
    coll = f"chunking_{name}"
    if coll in {c.name for c in qdrant.get_collections().collections}:
        qdrant.delete_collection(coll)
    qdrant.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    embeddings = vo.embed(chunks, model=MODEL, input_type="document").embeddings
    qdrant.upsert(
        collection_name=coll,
        points=[
            PointStruct(
                id=i,
                vector=v,
                payload={"text": t, "chunk_index": i},
            )
            for i, (t, v) in enumerate(zip(chunks, embeddings))
        ],
    )
    return coll


print("Indexing all three variants...")
coll_naive = index_chunks("naive", naive_chunks)
coll_recursive = index_chunks("recursive_no_overlap", recursive_chunks)
coll_overlap = index_chunks("recursive_with_overlap", overlap_chunks)
print("  ✓ Indexed.\n")

queries = [
    "How much is the equipment stipend for remote workers?",
    "What internet speed do I need at home?",
    "Can I use my personal laptop for work?",
    "Who is eligible for remote work?",
]


def search(coll: str, query: str, k: int = 1) -> tuple[float, str]:
    qvec = vo.embed([query], model=MODEL, input_type="query").embeddings[0]
    hits = qdrant.query_points(
        collection_name=coll, query=qvec, limit=k, with_payload=True
    ).points
    if not hits:
        return (0.0, "")
    return (hits[0].score, hits[0].payload["text"])


for q in queries:
    print(f"[bold]Q:[/bold] {q}\n")

    table = Table(show_lines=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Top retrieved chunk (truncated)", overflow="fold", max_width=70)

    for label, coll in [
        ("Naive", coll_naive),
        ("Recursive (no overlap)", coll_recursive),
        ("Recursive (with overlap)", coll_overlap),
    ]:
        score, text = search(coll, q)
        snippet = text[:200].replace("\n", " ")
        if len(text) > 200:
            snippet += "..."
        table.add_row(label, f"{score:.4f}", snippet)

    print(table)
    print()


# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Done"))
print("Three collections preserved for inspection:")
print("  - chunking_naive")
print("  - chunking_recursive_no_overlap")
print("  - chunking_recursive_with_overlap")
print("Visit http://localhost:6333/dashboard to compare them visually.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
