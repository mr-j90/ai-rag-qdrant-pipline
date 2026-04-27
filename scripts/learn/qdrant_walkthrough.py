"""
LlamaIndex × Qdrant walkthrough — learn the storage layer in isolation.

LlamaIndex hides a lot of Qdrant. That's the whole point of using it. But it's
still useful to see what's happening underneath, so this script walks through:

    1. Create a QdrantVectorStore (LlamaIndex's wrapper)
    2. Build a VectorStoreIndex from a few Documents
    3. Inspect what actually got stored in Qdrant under the hood
    4. Run a retrieval through the index
    5. Run a metadata-FILTERED retrieval (the killer feature)
    6. Drop down to the raw QdrantClient for admin operations
    7. Clean up

The lesson: LlamaIndex's index is the *front door*; QdrantClient is the
*service entrance*. Use the front door for reads/writes, the service
entrance for admin (count, reset, list distinct payload values).

Prereq:  docker compose up -d
Run:     uv run python -m scripts.learn.qdrant_walkthrough
"""
from __future__ import annotations

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rich import print
from rich.rule import Rule

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "qdrant_walkthrough"
settings = get_settings()

qdrant = QdrantClient(url=settings.qdrant_url)
embed_model = VoyageEmbedding(
    model_name=settings.voyage_model,
    voyage_api_key=settings.voyage_api_key,
)


# ---------------------------------------------------------------------------
# 1. Create the LlamaIndex-flavored vector store
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. QdrantVectorStore — LlamaIndex's adapter"))

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)
    print(f"Deleted old '{COLLECTION}' collection.")

vector_store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
print(f"Created QdrantVectorStore for collection '{COLLECTION}'.")
print("[dim]Note: the Qdrant collection itself is created lazily on the first write.[/dim]")


# ---------------------------------------------------------------------------
# 2. Build a VectorStoreIndex from Documents
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Index a few Documents"))

docs = [
    Document(text="Qdrant is a vector database written in Rust.",
             metadata={"source": "wiki.pdf", "page": 1, "topic": "infra"}),
    Document(text="FastAPI is a Python web framework.",
             metadata={"source": "fastapi.pdf", "page": 1, "topic": "web"}),
    Document(text="Voyage AI provides text embeddings.",
             metadata={"source": "voyage.pdf", "page": 3, "topic": "ml"}),
    Document(text="Claude is Anthropic's family of LLMs.",
             metadata={"source": "anthropic.pdf", "page": 2, "topic": "ml"}),
    Document(text="RAG combines retrieval with generation.",
             metadata={"source": "rag.pdf", "page": 1, "topic": "ml"}),
    Document(text="Cosine similarity measures angle between vectors.",
             metadata={"source": "math.pdf", "page": 7, "topic": "math"}),
]

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
)
print(f"Indexed {len(docs)} documents. Collection count: "
      f"{qdrant.count(COLLECTION, exact=True).count}")


# ---------------------------------------------------------------------------
# 3. Peek under the hood — what did LlamaIndex actually write?
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. What got stored?"))

raw_points, _ = qdrant.scroll(
    collection_name=COLLECTION, limit=2, with_payload=True, with_vectors=False
)
for p in raw_points:
    print(f"  point id: {p.id}")
    # LlamaIndex stores text under `_node_content` and metadata under `metadata`.
    keys = sorted((p.payload or {}).keys())
    print(f"  payload keys: {keys}")
    md = (p.payload or {}).get("metadata") or {}
    print(f"  metadata: {md}")


# ---------------------------------------------------------------------------
# 4. Retrieval through the index (the front door)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Retrieve through the index"))

retriever = index.as_retriever(similarity_top_k=3)
hits = retriever.retrieve("What's a good database for embeddings?")
for h in hits:
    print(f"  score={h.score:+.4f}  topic={h.node.metadata.get('topic')}  "
          f"text='{h.node.get_content()}'")


# ---------------------------------------------------------------------------
# 5. Metadata-filtered retrieval — the killer feature
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Filtered retrieval"))

ml_only = MetadataFilters(filters=[MetadataFilter(key="topic", value="ml")])
filtered = index.as_retriever(similarity_top_k=3, filters=ml_only).retrieve(
    "Tell me about machine learning"
)
print("Filter: topic == 'ml'")
for h in filtered:
    print(f"  score={h.score:+.4f}  topic={h.node.metadata.get('topic')}  "
          f"text='{h.node.get_content()}'")


# ---------------------------------------------------------------------------
# 6. Service entrance — raw QdrantClient for admin
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Raw QdrantClient for admin operations"))

# LlamaIndex doesn't expose collection count, payload listing, etc.
# That's fine — drop down to qdrant_client when you need it.
print(f"Collection count: {qdrant.count(COLLECTION, exact=True).count}")
info = qdrant.get_collection(COLLECTION)
print(f"Vector size: {info.config.params.vectors.size}")
print(f"Distance:    {info.config.params.vectors.distance}")


# ---------------------------------------------------------------------------
# 7. Cleanup
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Cleanup"))
print(f"Collection '{COLLECTION}' kept around. http://localhost:6333/dashboard to explore.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
