"""
LangChain × Qdrant walkthrough — learn the storage layer in isolation.

LangChain hides a lot of Qdrant. That's the whole point of using it. But it's
still useful to see what's happening underneath, so this script walks through:

    1. Create a QdrantVectorStore (LangChain's wrapper)
    2. Add a few Documents
    3. Inspect what actually got stored in Qdrant under the hood
    4. Run a similarity search
    5. Run a metadata-FILTERED search (the killer feature)
    6. Use as_retriever() — the Runnable interface
    7. Drop down to the raw QdrantClient for admin operations
    8. Clean up

The lesson: LangChain's vector store is the *front door*; QdrantClient is the
*service entrance*. Use the front door for reads/writes, the service
entrance for admin (count, reset, list distinct payload values).

Prereq:  docker compose up -d
Run:     uv run python -m scripts.learn.qdrant_walkthrough
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)
from rich import print
from rich.rule import Rule

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "qdrant_walkthrough"
settings = get_settings()

qdrant = QdrantClient(url=settings.qdrant_url)
embeddings = VoyageAIEmbeddings(
    model=settings.voyage_model,
    api_key=settings.voyage_api_key,
)


# ---------------------------------------------------------------------------
# 1. Create the LangChain-flavored vector store
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. QdrantVectorStore — LangChain's adapter"))

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)
    print(f"Deleted old '{COLLECTION}' collection.")

# QdrantVectorStore needs the collection to exist already (it doesn't auto-create
# unless you go through `.from_documents`). Create it explicitly so we can show
# both flows side by side.
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION,
    embedding=embeddings,
)
print(f"Created QdrantVectorStore for collection '{COLLECTION}'.")
print("[dim]The store is just a thin object — connection + collection name + embedding fn.[/dim]")


# ---------------------------------------------------------------------------
# 2. Add Documents
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Add a few Documents"))

docs = [
    Document(page_content="Qdrant is a vector database written in Rust.",
             metadata={"source": "wiki.pdf", "page": 1, "topic": "infra"}),
    Document(page_content="FastAPI is a Python web framework.",
             metadata={"source": "fastapi.pdf", "page": 1, "topic": "web"}),
    Document(page_content="Voyage AI provides text embeddings.",
             metadata={"source": "voyage.pdf", "page": 3, "topic": "ml"}),
    Document(page_content="Claude is Anthropic's family of LLMs.",
             metadata={"source": "anthropic.pdf", "page": 2, "topic": "ml"}),
    Document(page_content="RAG combines retrieval with generation.",
             metadata={"source": "rag.pdf", "page": 1, "topic": "ml"}),
    Document(page_content="Cosine similarity measures angle between vectors.",
             metadata={"source": "math.pdf", "page": 7, "topic": "math"}),
]

ids = vector_store.add_documents(docs)
print(f"Added {len(ids)} documents. Collection count: "
      f"{qdrant.count(COLLECTION, exact=True).count}")


# ---------------------------------------------------------------------------
# 3. Peek under the hood — what did LangChain actually write?
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. What got stored?"))

raw_points, _ = qdrant.scroll(
    collection_name=COLLECTION, limit=2, with_payload=True, with_vectors=False
)
for p in raw_points:
    print(f"  point id: {p.id}")
    keys = sorted((p.payload or {}).keys())
    print(f"  payload keys: {keys}")
    md = (p.payload or {}).get("metadata") or {}
    print(f"  metadata: {md}")
print("[dim]LangChain stores `page_content` + `metadata` as nested keys, plus its own "
      "internal fields. That's why filters use `metadata.<key>` paths.[/dim]")


# ---------------------------------------------------------------------------
# 4. Similarity search
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Similarity search (returns Documents)"))

hits = vector_store.similarity_search_with_score(
    "What's a good database for embeddings?", k=3
)
for doc, score in hits:
    print(f"  score={score:+.4f}  topic={doc.metadata.get('topic')}  "
          f"text='{doc.page_content}'")


# ---------------------------------------------------------------------------
# 5. Filtered search — the killer feature
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Filtered search via qdrant Filter objects"))

ml_only = Filter(
    must=[FieldCondition(key="metadata.topic", match=MatchValue(value="ml"))]
)
filtered = vector_store.similarity_search_with_score(
    "Tell me about machine learning", k=3, filter=ml_only
)
print("Filter: metadata.topic == 'ml'")
for doc, score in filtered:
    print(f"  score={score:+.4f}  topic={doc.metadata.get('topic')}  "
          f"text='{doc.page_content}'")


# ---------------------------------------------------------------------------
# 6. as_retriever() — the Runnable interface
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. as_retriever() — plug into LCEL chains"))

retriever = vector_store.as_retriever(search_kwargs={"k": 2, "filter": ml_only})
# Retrievers ARE Runnables: .invoke / .batch / .stream all work.
docs_only = retriever.invoke("Tell me about LLMs")
for d in docs_only:
    print(f"  {d.page_content}  ({d.metadata.get('source')})")
print("[dim]The retriever drops scores — use similarity_search_with_score if you need them.[/dim]")


# ---------------------------------------------------------------------------
# 7. Service entrance — raw QdrantClient for admin
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Raw QdrantClient for admin operations"))

# LangChain doesn't expose collection count, payload listing, etc.
# Drop down to qdrant_client when you need it.
print(f"Collection count: {qdrant.count(COLLECTION, exact=True).count}")
info = qdrant.get_collection(COLLECTION)
print(f"Vector size: {info.config.params.vectors.size}")
print(f"Distance:    {info.config.params.vectors.distance}")


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]8. Cleanup"))
print(f"Collection '{COLLECTION}' kept around. http://localhost:6333/dashboard to explore.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
