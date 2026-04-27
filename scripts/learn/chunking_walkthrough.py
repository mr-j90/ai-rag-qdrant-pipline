"""
LlamaIndex chunking walkthrough — node parsers, side by side.

LlamaIndex calls chunkers "node parsers." They all take Documents and emit
a list of Nodes (text + metadata). The interesting ones for prose:

    - TokenTextSplitter         — fixed token windows, ignores boundaries
    - SentenceSplitter          — respects sentence + paragraph boundaries
                                  (this is what `IngestionPipeline` uses by default
                                  in our project, configured by chunk_size/overlap)
    - SemanticSplitterNodeParser — splits where embedding similarity drops

This walkthrough takes one document, runs each parser on it, indexes each
variant into its own Qdrant collection, and runs the same query against
each. You can see retrieval quality change.

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY in .env

Run:
    uv run python -m scripts.learn.chunking_walkthrough
"""
from __future__ import annotations

import textwrap

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rich import print
from rich.rule import Rule
from rich.table import Table

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
settings = get_settings()
embed_model = VoyageEmbedding(
    model_name=settings.voyage_model,
    voyage_api_key=settings.voyage_api_key,
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

source_doc = Document(text=DOCUMENT, metadata={"source": "remote_work_policy.md"})


# ---------------------------------------------------------------------------
# 1. TokenTextSplitter — the closest thing to "naive fixed window"
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. TokenTextSplitter (fixed token windows, no overlap)"))

token_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
token_nodes = token_splitter.get_nodes_from_documents([source_doc])
print(f"Got {len(token_nodes)} nodes.\n")
for i, n in enumerate(token_nodes):
    text = n.get_content()
    print(f"[bold]Node {i}[/bold] ({len(text)} chars):")
    print(f"  start: ...{text[:60]!r}")
    print(f"  end:   ...{text[-60:]!r}\n")
print("[dim]Token windows ignore section headers and sentence boundaries.[/dim]")


# ---------------------------------------------------------------------------
# 2. SentenceSplitter, no overlap
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. SentenceSplitter, no overlap"))

sentence_no_overlap = SentenceSplitter(chunk_size=128, chunk_overlap=0)
no_overlap_nodes = sentence_no_overlap.get_nodes_from_documents([source_doc])
print(f"Got {len(no_overlap_nodes)} nodes.\n")
for i, n in enumerate(no_overlap_nodes):
    text = n.get_content()
    print(f"[bold]Node {i}[/bold] ({len(text)} chars):")
    print(f"  start: {text[:60]!r}...")
    print(f"  end:   ...{text[-60:]!r}\n")
print("[dim]Better — sentences and paragraphs stay intact. But chunk seams are still hard cuts.[/dim]")


# ---------------------------------------------------------------------------
# 3. SentenceSplitter WITH overlap — the production default
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. SentenceSplitter WITH overlap (production default)"))

sentence_overlap = SentenceSplitter(chunk_size=128, chunk_overlap=24)
overlap_nodes = sentence_overlap.get_nodes_from_documents([source_doc])
print(f"Got {len(overlap_nodes)} nodes.\n")
for i in range(len(overlap_nodes) - 1):
    end = overlap_nodes[i].get_content()[-60:]
    start = overlap_nodes[i + 1].get_content()[:60]
    print(f"[bold]Node {i} → {i + 1} boundary[/bold]")
    print(f"  end of {i}:    ...{end!r}")
    print(f"  start of {i + 1}:  {start!r}...\n")
print("[dim]Deliberate duplication at boundaries: a fact near the seam appears in both nodes, "
      "so retrieval can't lose it.[/dim]")


# ---------------------------------------------------------------------------
# 4. SemanticSplitterNodeParser — split where meaning shifts
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. SemanticSplitterNodeParser (embedding-aware boundaries)"))

semantic_splitter = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95,
    buffer_size=1,
)
semantic_nodes = semantic_splitter.get_nodes_from_documents([source_doc])
print(f"Got {len(semantic_nodes)} nodes.\n")
for i, n in enumerate(semantic_nodes):
    text = n.get_content()
    snippet = text[:120].replace("\n", " ")
    print(f"[bold]Node {i}[/bold] ({len(text)} chars): {snippet}...")
print("\n[dim]Splits happen where consecutive sentences become semantically dissimilar — "
      "tends to align with section boundaries when the doc is well-structured.[/dim]")


# ---------------------------------------------------------------------------
# 5. Index each variant and compare retrieval
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Compare retrieval across strategies"))


def index_nodes(name: str, nodes: list) -> VectorStoreIndex:
    coll = f"chunking_{name}"
    if coll in {c.name for c in qdrant.get_collections().collections}:
        qdrant.delete_collection(coll)
    vs = QdrantVectorStore(client=qdrant, collection_name=coll)
    return VectorStoreIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=embed_model,
    )


print("Indexing all four variants...")
idx_token = index_nodes("token", token_nodes)
idx_no_overlap = index_nodes("sentence_no_overlap", no_overlap_nodes)
idx_overlap = index_nodes("sentence_with_overlap", overlap_nodes)
idx_semantic = index_nodes("semantic", semantic_nodes)
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

    for label, idx in [
        ("Token", idx_token),
        ("Sentence (no overlap)", idx_no_overlap),
        ("Sentence (with overlap)", idx_overlap),
        ("Semantic", idx_semantic),
    ]:
        hits = idx.as_retriever(similarity_top_k=1).retrieve(q)
        if not hits:
            table.add_row(label, "-", "(no hits)")
            continue
        text = hits[0].node.get_content()
        snippet = text[:200].replace("\n", " ")
        if len(text) > 200:
            snippet += "..."
        table.add_row(label, f"{hits[0].score:.4f}", snippet)
    print(table)
    print()


# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Done"))
print("Collections preserved:")
print("  - chunking_token")
print("  - chunking_sentence_no_overlap")
print("  - chunking_sentence_with_overlap")
print("  - chunking_semantic")
print("Compare them at http://localhost:6333/dashboard.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
