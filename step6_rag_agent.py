"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 6 — RAG (Retrieval-Augmented Generation)              ║
╚══════════════════════════════════════════════════════════════╝

WHAT IS RAG?
  RAG = Retrieval-Augmented Generation.

  The problem it solves:
    LLMs only know what was in their training data. Your own docs,
    internal wikis, and domain knowledge aren't there. RAG lets the
    agent search YOUR knowledge base and use those results to answer.

  How it works in two phases:

    INDEXING (once at startup):
      documents → split into chunks → embed each chunk → store in vector DB

    RETRIEVAL (at query time):
      question → embed question → find similar chunks in DB → return top-k

  "Similar" means similar MEANING, not just matching keywords.
  This works because embeddings place semantically related text
  close together in vector space.

KEY NEW CONCEPTS:

  Embeddings
    Converts text → a list of floats (a vector) that captures meaning.
    "token bucket" and "token refill" will have similar vectors.
    We use HuggingFace's all-MiniLM-L6-v2 (free, runs locally, ~90MB).

  Vector Store (FAISS)
    A database that stores embedding vectors and can find the nearest
    ones to a query vector extremely fast. We use FAISS — it runs
    entirely in memory, no server needed.

  Text Splitter
    Long documents are cut into smaller overlapping chunks so retrieval
    returns a focused paragraph, not an entire document.

  RAG as a Tool
    We wrap the retriever as a @tool so the agent can choose when to
    search the knowledge base, just like any other tool.

WHAT'S NEW vs STEP 5:
  Step 5 had 3 hardcoded tools with fixed answers.
  Step 6 adds a 4th tool — search_knowledge_base — that does semantic
  search over a rich document collection at runtime.
  The agent now answers from RETRIEVED CONTEXT, not hardcoded strings.

NOTE: First run downloads the embedding model (~90MB). After that it's cached.

HOW TO RUN:
  uv pip install faiss-cpu langchain-huggingface sentence-transformers langchain-text-splitters
  export GROQ_API_KEY=your_key_here
  uv run step6_rag_agent.py
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS  # noqa: F401 — community still owns FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

print("=" * 60)
print("STEP 6: RAG Agent")
print("=" * 60)


# ── 1. Knowledge Base ─────────────────────────────────────────────────────────
#
# These are the documents the agent will retrieve from.
# In a real app these could be PDFs, web pages, database records, etc.
# Here we embed them directly as strings for clarity.
#
DOCUMENTS = [
    Document(
        page_content="""
        Token Bucket Algorithm — Detailed Guide

        The Token Bucket is the most widely used rate limiting algorithm in production
        systems, particularly for APIs and network bandwidth management.

        How it works:
        - A "bucket" holds tokens up to a maximum capacity.
        - Tokens are added at a constant refill rate (e.g., 10 tokens/second).
        - Each incoming request consumes one token.
        - If the bucket has tokens, the request is allowed and a token is removed.
        - If the bucket is empty, the request is rejected (HTTP 429).

        Burst handling:
        When traffic is low, tokens accumulate up to the bucket capacity.
        This stored capacity allows short bursts above the average rate.
        For example: capacity=100, refill=10/s — after 10 idle seconds
        the bucket is full and can absorb a burst of 100 requests instantly.

        Parameters:
        - capacity: maximum tokens (= maximum burst size)
        - refill_rate: tokens added per second (= sustained rate)

        Best for: APIs, microservices, any system where occasional bursts are acceptable.
        """,
        metadata={"source": "token_bucket_guide", "algorithm": "token_bucket"}
    ),
    Document(
        page_content="""
        Fixed Window Counter — Detailed Guide

        The Fixed Window algorithm divides time into discrete, non-overlapping windows
        (e.g., every 60 seconds). Each window has an independent counter.

        How it works:
        - Track the current window start time and a request counter.
        - When a request arrives, check if we're still in the same window.
        - If yes: increment counter. Allow if counter <= limit, else reject.
        - If no: reset counter to 1, update window start. Allow request.

        The boundary spike problem:
        A client can send limit requests at 11:59:59 and another limit requests
        at 12:00:01 — that's 2× the limit in just 2 seconds, because they straddle
        two windows. This is the main weakness of Fixed Window.

        Parameters:
        - limit: max requests per window
        - window_size: duration of each window (e.g., 60 seconds)

        Best for: simple use cases where the boundary spike is acceptable,
        or when memory efficiency is critical.
        """,
        metadata={"source": "fixed_window_guide", "algorithm": "fixed_window"}
    ),
    Document(
        page_content="""
        Sliding Window Log — Detailed Guide

        The Sliding Window Log keeps a timestamped log of every allowed request.
        It provides the most precise rate limiting with no boundary effects.

        How it works:
        - Maintain a sorted list (or deque) of timestamps of allowed requests.
        - On each new request:
          1. Remove all timestamps older than now - window_size.
          2. Count remaining timestamps.
          3. If count < limit: add current timestamp, allow request.
          4. If count >= limit: reject request.

        Memory usage:
        Stores one timestamp per allowed request. If your limit is 1000 req/min,
        the log holds up to 1000 entries. This is O(limit) memory per client.
        For high-traffic systems with many clients, this adds up.

        Precision:
        This is the only algorithm with zero boundary effects. The window
        truly "slides" — it always looks back exactly window_size seconds.

        Parameters:
        - limit: max requests in any window_size duration
        - window_size: the sliding window duration

        Best for: systems requiring precise enforcement, security-sensitive rate limiting.
        """,
        metadata={"source": "sliding_window_log_guide", "algorithm": "sliding_window_log"}
    ),
    Document(
        page_content="""
        Leaky Bucket Algorithm — Detailed Guide

        The Leaky Bucket models a bucket with a hole — water (requests) flows in
        at any rate but leaks out at a constant rate. It enforces a perfectly
        smooth output regardless of bursty input.

        How it works:
        - Maintain a queue of pending requests with maximum size = capacity.
        - Requests are processed ("leaked") at a fixed rate.
        - New requests are added to the queue if space is available.
        - New requests are rejected if the queue is full.

        Output smoothing:
        Even if 100 requests arrive simultaneously, they are processed one
        by one at the leak rate. The output is always perfectly smooth.
        There is NO burst capability — every request waits in line.

        Comparison with Token Bucket:
        - Token Bucket: allows bursts (stored tokens), variable output rate
        - Leaky Bucket:  no bursts allowed, constant output rate
        Use Leaky Bucket when the downstream system is fragile and cannot
        handle any spikes at all.

        Parameters:
        - capacity: max queue size (requests that can wait)
        - leak_rate: requests processed per second

        Best for: protecting fragile downstream services, network traffic shaping.
        """,
        metadata={"source": "leaky_bucket_guide", "algorithm": "leaky_bucket"}
    ),
    Document(
        page_content="""
        Distributed Rate Limiting

        Single-server rate limiting is straightforward, but most production systems
        run multiple instances behind a load balancer. This creates challenges.

        The problem:
        If you have 3 servers each allowing 100 req/s, a client can hit
        all 3 servers for a total of 300 req/s — 3× your intended limit.

        Solutions:

        1. Centralized store (Redis):
           All servers read and write to a shared Redis instance.
           Use atomic Redis commands (INCR, SET with TTL, Lua scripts)
           to avoid race conditions.
           Pros: accurate, consistent. Cons: Redis becomes a bottleneck/SPOF.

        2. Sticky sessions:
           Route each client always to the same server.
           Pros: no coordination needed. Cons: uneven load distribution.

        3. Approximate counting:
           Each server tracks locally, periodically syncs to a central store.
           Accept slight inaccuracy (~5-10%) in exchange for performance.
           Good enough for most rate limiting use cases.

        Redis data structures for rate limiting:
        - Fixed Window: INCR key, EXPIRE key window_size
        - Sliding Window: ZADD with timestamp as score, ZREMRANGEBYSCORE
        - Token Bucket: Lua script to atomically check and decrement
        """,
        metadata={"source": "distributed_rate_limiting", "algorithm": "distributed"}
    ),
    Document(
        page_content="""
        Rate Limiting Best Practices for APIs

        Response headers:
        Always include rate limit info in response headers so clients can adapt:
          X-RateLimit-Limit: 100
          X-RateLimit-Remaining: 73
          X-RateLimit-Reset: 1640995200

        HTTP 429 Too Many Requests:
        Return 429 when a request is rejected. Include Retry-After header
        with the number of seconds until the client can retry.

        Per-user vs global limits:
        Apply limits per API key or user ID, not just per IP.
        IPs can be shared (NAT, corporate proxies) causing false positives.

        Tiered limits:
        Different user tiers get different limits:
          - Free tier:    10 req/min
          - Pro tier:    100 req/min
          - Enterprise:  unlimited or custom

        Graceful degradation:
        When a client hits the limit, consider returning a cached response
        or a degraded response instead of a hard error. Improves UX.

        Burst allowance:
        Token Bucket is ideal for APIs because it allows legitimate short
        bursts (e.g., a user clicking rapidly) while still enforcing
        long-term average limits.

        Algorithm recommendation by use case:
        - Public REST API:       Token Bucket (burst tolerance)
        - Internal microservice: Fixed Window (simplicity, low overhead)
        - Security/auth endpoint: Sliding Window Log (precise, no bypass)
        - Video streaming:       Leaky Bucket (smooth bandwidth)
        """,
        metadata={"source": "api_best_practices", "algorithm": "general"}
    ),
]


# ── 2. Build the Vector Store ─────────────────────────────────────────────────
#
# Step 1: Split documents into chunks.
# Long documents are split so retrieval returns a focused excerpt.
# chunk_overlap keeps context across chunk boundaries.
#
print("\n[Building knowledge base...]")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk is at most 500 characters
    chunk_overlap=50,    # 50-char overlap between consecutive chunks
)
chunks = splitter.split_documents(DOCUMENTS)
print(f"  {len(DOCUMENTS)} documents → {len(chunks)} chunks after splitting")

# Step 2: Create embeddings.
# all-MiniLM-L6-v2 is a small (22M param), fast, accurate embedding model.
# First run downloads ~90MB. After that it's cached in ~/.cache/huggingface/.
#
print("  Loading embedding model (downloads ~90MB on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Build FAISS vector store.
# FAISS.from_documents() embeds every chunk and stores the vectors in memory.
# The index is ready for similarity search instantly.
#
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"  Vector store ready — {len(chunks)} vectors indexed")

# Step 4: Create a retriever.
# .as_retriever() wraps the vector store in a standard Retriever interface.
# search_kwargs={"k": 3} means "return the 3 most similar chunks".
#
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ── 3. RAG Tool ───────────────────────────────────────────────────────────────
#
# Wrap the retriever as a @tool so the agent can call it.
# When called, it embeds the query and returns the top matching chunks.
#
@tool
def search_knowledge_base(query: str) -> str:
    """Searches the rate limiting knowledge base for detailed information.
    Use this for in-depth questions about algorithms, distributed systems,
    best practices, implementation details, or API design.
    This tool retrieves from a rich documentation corpus.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    # Format the retrieved chunks into a readable string for the LLM
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(results)


# ── 4. Remaining tools (same as step 5) ──────────────────────────────────────

@tool
def get_algorithm_info(algorithm: str) -> str:
    """Returns a brief description of a rate limiting algorithm.
    Valid values: token_bucket, fixed_window, sliding_window_log, leaky_bucket
    """
    info = {
        "token_bucket":       "Tokens refill at a fixed rate. Allows bursts up to capacity.",
        "fixed_window":       "Counter resets every window. Simple but has boundary spike risk.",
        "sliding_window_log": "Logs every request timestamp. Precise, no spikes, O(limit) memory.",
        "leaky_bucket":       "Requests drain at constant rate. Perfectly smooth output.",
    }
    key = algorithm.lower().replace(" ", "_")
    return info.get(key, f"Unknown algorithm '{algorithm}'.")


@tool
def recommend_algorithm(requirements: str) -> str:
    """Recommends an algorithm given a description of requirements."""
    r = requirements.lower()
    if any(w in r for w in ["burst", "spike", "peak"]):
        return "Token Bucket — tolerates short bursts while enforcing average rate."
    if any(w in r for w in ["smooth", "even", "constant"]):
        return "Leaky Bucket — enforces perfectly smooth output, zero bursts."
    if any(w in r for w in ["simple", "memory", "minimal"]):
        return "Fixed Window — simplest implementation, O(1) memory."
    if any(w in r for w in ["precise", "accurate", "exact", "security"]):
        return "Sliding Window Log — most precise, no boundary effects."
    return "Token Bucket — solid default: simple, burst-tolerant, O(1) memory."


@tool
def calculate_token_bucket(requests_per_second: str, burst_size: str) -> str:
    """Calculates Token Bucket config for a given rate and burst size."""
    rps   = float(requests_per_second)
    burst = int(burst_size)
    return (
        f"Token Bucket config:\n"
        f"  capacity        = {burst} tokens\n"
        f"  refill rate     = {rps} tokens/sec\n"
        f"  refill interval = {round(1000/rps, 2)} ms per token"
    )


# ── 5. Agent (same structure as step 5, now with 4 tools) ────────────────────

tools = [search_knowledge_base, get_algorithm_info, recommend_algorithm, calculate_token_bucket]

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Rate Limiting Expert assistant with access to a knowledge base. "
    "For detailed or in-depth questions, ALWAYS use search_knowledge_base first "
    "to retrieve accurate information before answering. "
    "Combine retrieved context with your tools to give thorough answers."
))

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_llm(state: AgentState) -> dict:
    response = model_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    if response.tool_calls:
        print(f"  [llm] → {len(response.tool_calls)} tool call(s):")
        for tc in response.tool_calls:
            print(f"         {tc['name']}({tc['args']})")
    else:
        print("  [llm] → final answer ready")
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("llm",   call_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
agent = graph.compile()


# ── 6. Run the agent ──────────────────────────────────────────────────────────

def chat(question: str):
    print(f"\n{'─'*60}")
    print(f"User: {question}")
    print(f"{'─'*60}")
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\nAssistant: {result['messages'][-1].content}")


# These questions require deep knowledge from the knowledge base —
# the simple tools from step 5 wouldn't have these answers.
chat("What is the boundary spike problem in Fixed Window and how bad can it get?")

chat("How do I implement rate limiting across multiple servers? What are my options?")

chat("What HTTP headers should I return when rate limiting an API?")

chat("For a video streaming service, which algorithm should I use and why?")

print("\n✓ Step 6 complete — the agent now answers from your own knowledge base.")
print("\n  What's new vs step 5:")
print("  - Documents embedded and indexed at startup")
print("  - search_knowledge_base tool does semantic search at query time")
print("  - Agent retrieves relevant context before answering")
