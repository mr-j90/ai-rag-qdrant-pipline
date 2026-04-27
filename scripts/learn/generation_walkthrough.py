"""
LangChain generation walkthrough — close the RAG loop with LCEL.

The pieces:
    - QdrantVectorStore       — already familiar from earlier walkthroughs
    - ChatAnthropic           — LangChain's wrapper around the Claude API
    - ChatPromptTemplate      — the prompt
    - StrOutputParser         — extracts the .content string from the AIMessage
    - LCEL pipe operator (|)  — composes Runnables into a chain

We'll:
    1. Build a small RAG corpus (Document → split → embed → index)
    2. Ask Claude WITHOUT context (baseline)
    3. Build an LCEL chain: prompt | llm | parser, run it grounded
    4. Add the retriever to the chain so it auto-fetches context
    5. The honesty test — out-of-scope question
    6. Stream a response token-by-token

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY and ANTHROPIC_API_KEY in .env

Run:
    uv run python -m scripts.learn.generation_walkthrough
"""
from __future__ import annotations

import textwrap

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print
from rich.panel import Panel
from rich.rule import Rule

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "generation_walkthrough"
settings = get_settings()

embeddings = VoyageAIEmbeddings(
    model=settings.voyage_model,
    api_key=settings.voyage_api_key,
)
llm = ChatAnthropic(
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
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
)

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(
    [Document(page_content=DOCUMENT, metadata={"source": "remote_work_policy.md"})]
)
print(f"Split into {len(chunks)} chunks.")

vector_store = QdrantVectorStore(
    client=qdrant, collection_name=COLLECTION, embedding=embeddings
)
vector_store.add_documents(chunks)
print("Indexed.\n")


# ---------------------------------------------------------------------------
# 1. Naked Claude (no context)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Ask Claude WITHOUT any context"))

question = "What is the equipment stipend for remote workers at Acme Corp?"
print(f"[bold]Question:[/bold] {question}\n")

# Even at this level, ChatAnthropic IS a Runnable, so .invoke works.
naked_msg = llm.invoke(question)
print(Panel(naked_msg.content, title="Naked Claude", border_style="yellow"))
print("[dim]Claude has no idea about Acme Corp's specific policies.[/dim]\n")


# ---------------------------------------------------------------------------
# 2. The LCEL chain — prompt | llm | parser
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Build a basic LCEL chain"))

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Answer using ONLY the provided context. "
            "Each passage is labeled like [#1], [#2]. Cite passages inline with the same labels. "
            "If the context doesn't contain the answer, say so honestly.",
        ),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

chain = PROMPT | llm | StrOutputParser()
print("[dim]chain = prompt | llm | parser  (every | is a Runnable composition).[/dim]\n")


# ---------------------------------------------------------------------------
# 3. Grounded answer — manual retrieval, manual context formatting
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. Grounded RAG (manual retrieve → format → run chain)"))


def format_docs(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        blocks.append(
            f"[#{i}] (source: {meta.get('source', 'unknown')}, page: {meta.get('page', '?')})\n"
            f"{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


pairs = vector_store.similarity_search_with_score(question, k=3)
hits = [doc for doc, _ in pairs]
print(f"Retrieved {len(hits)} chunks. Top scores: {[round(s, 3) for _, s in pairs]}\n")

answer = chain.invoke({"context": format_docs(hits), "question": question})
print(Panel(answer, title="Grounded Claude", border_style="green"))


# ---------------------------------------------------------------------------
# 4. Composed RAG chain — retriever folded in
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Compose the retriever into the chain"))

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# A composed RAG chain. The trick: `RunnablePassthrough.assign` keeps the
# original input dict (which has `question`) and adds new keys to it.
# We compute `context` from the retriever, then route the dict into the chain.
rag_chain = (
    RunnablePassthrough.assign(
        context=(lambda x: x["question"]) | retriever | format_docs
    )
    | chain
)

q2 = "What's the application process for remote work?"
print(f"[bold]Question:[/bold] {q2}\n")
out = rag_chain.invoke({"question": q2})
print(Panel(out, title="Composed RAG chain", border_style="green"))


# ---------------------------------------------------------------------------
# 5. The honesty test
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. The honesty test — out-of-scope question"))

oos_question = "What's Acme Corp's parental leave policy?"
print(f"[bold]Question:[/bold] {oos_question}")
print("[dim](This document doesn't say anything about parental leave.)[/dim]\n")

oos_answer = rag_chain.invoke({"question": oos_question})
print(Panel(oos_answer, title="Grounded Claude (out-of-scope)", border_style="red"))
print("[dim]A well-grounded RAG should say 'I don't know' rather than confabulate.[/dim]\n")


# ---------------------------------------------------------------------------
# 6. Streaming
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Streaming response (.stream() works on any Runnable)"))

stream_question = "Summarize the security requirements for remote workers."
print(f"[bold]Question:[/bold] {stream_question}\n")
print("[bold]Streamed answer:[/bold]")

for chunk in rag_chain.stream({"question": stream_question}):
    print(chunk, end="", flush=True)
print("\n")
print("[dim]Same chain, different method — `.stream()` returns an iterator of partial outputs. "
      "Plug into FastAPI's StreamingResponse for SSE.[/dim]\n")


# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Done"))
print(f"Collection '{COLLECTION}' preserved at http://localhost:6333/dashboard")
print("[bold green]✓ Walkthrough complete. End-to-end RAG via LangChain LCEL.[/bold green]")
