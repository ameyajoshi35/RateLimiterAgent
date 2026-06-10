"""
RAG agent extracted from step6 — imported by the FastAPI server.
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── Knowledge Base ─────────────────────────────────────────────────────────────

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
        metadata={"source": "token_bucket_guide"}
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
        at 12:00:01 — that's 2x the limit in just 2 seconds, because they straddle
        two windows. This is the main weakness of Fixed Window.

        Best for: simple use cases where the boundary spike is acceptable,
        or when memory efficiency is critical.
        """,
        metadata={"source": "fixed_window_guide"}
    ),
    Document(
        page_content="""
        Sliding Window Log — Detailed Guide

        The Sliding Window Log keeps a timestamped log of every allowed request.
        It provides the most precise rate limiting with no boundary effects.

        How it works:
        - Maintain a sorted list of timestamps of allowed requests.
        - On each new request:
          1. Remove all timestamps older than now - window_size.
          2. Count remaining timestamps.
          3. If count < limit: add current timestamp, allow request.
          4. If count >= limit: reject request.

        Memory usage: O(limit) per client.

        Best for: systems requiring precise enforcement, security-sensitive rate limiting.
        """,
        metadata={"source": "sliding_window_log_guide"}
    ),
    Document(
        page_content="""
        Leaky Bucket Algorithm — Detailed Guide

        The Leaky Bucket models a bucket with a hole — requests flow in at any rate
        but leak out at a constant rate, enforcing perfectly smooth output.

        How it works:
        - Maintain a queue of pending requests with maximum size = capacity.
        - Requests are processed at a fixed rate.
        - New requests are added to the queue if space is available.
        - New requests are rejected if the queue is full.

        Comparison with Token Bucket:
        - Token Bucket: allows bursts (stored tokens), variable output rate
        - Leaky Bucket: no bursts allowed, constant output rate

        Best for: protecting fragile downstream services, network traffic shaping.
        """,
        metadata={"source": "leaky_bucket_guide"}
    ),
    Document(
        page_content="""
        Distributed Rate Limiting

        The problem:
        If you have 3 servers each allowing 100 req/s, a client can hit
        all 3 servers for a total of 300 req/s — 3x your intended limit.

        Solutions:

        1. Centralized store (Redis):
           All servers read and write to a shared Redis instance.
           Use atomic Redis commands to avoid race conditions.
           Pros: accurate. Cons: Redis becomes a bottleneck/SPOF.

        2. Sticky sessions:
           Route each client always to the same server.
           Pros: no coordination. Cons: uneven load distribution.

        3. Approximate counting:
           Each server tracks locally, periodically syncs to Redis.
           Accept ~5-10% inaccuracy in exchange for performance.

        Redis data structures:
        - Fixed Window: INCR key, EXPIRE key window_size
        - Sliding Window: ZADD with timestamp as score, ZREMRANGEBYSCORE
        - Token Bucket: Lua script to atomically check and decrement
        """,
        metadata={"source": "distributed_rate_limiting"}
    ),
    Document(
        page_content="""
        Rate Limiting Best Practices for APIs

        Response headers — always include:
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
          Free tier:   10 req/min
          Pro tier:   100 req/min
          Enterprise: custom

        Algorithm recommendation by use case:
        - Public REST API:        Token Bucket (burst tolerance)
        - Internal microservice:  Fixed Window (simplicity)
        - Security/auth endpoint: Sliding Window Log (precise)
        - Video streaming:        Leaky Bucket (smooth bandwidth)
        """,
        metadata={"source": "api_best_practices"}
    ),
]

# ── Vector Store ───────────────────────────────────────────────────────────────

print("[Agent] Building knowledge base...")
splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks     = splitter.split_documents(DOCUMENTS)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever    = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"[Agent] {len(chunks)} chunks indexed.")

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """Searches the rate limiting knowledge base for detailed information.
    Use this for in-depth questions about algorithms, distributed systems,
    best practices, or implementation details.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[Source {i}: {doc.metadata.get('source')}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(results)


@tool
def get_algorithm_info(algorithm: str) -> str:
    """Returns a brief description of a rate limiting algorithm.
    Valid values: token_bucket, fixed_window, sliding_window_log, leaky_bucket
    """
    info = {
        "token_bucket":       "Tokens refill at a fixed rate up to a capacity cap. Allows bursts.",
        "fixed_window":       "Counter resets every window. Simple but has boundary spike risk.",
        "sliding_window_log": "Logs every request timestamp. Precise, no spikes, O(limit) memory.",
        "leaky_bucket":       "Requests drain at constant rate. Perfectly smooth output, no bursts.",
    }
    key = algorithm.lower().replace(" ", "_")
    return info.get(key, f"Unknown algorithm '{algorithm}'.")


@tool
def recommend_algorithm(requirements: str) -> str:
    """Recommends a rate limiting algorithm given a description of requirements."""
    r = requirements.lower()
    if any(w in r for w in ["burst", "spike", "peak"]):
        return "Token Bucket — tolerates short bursts while enforcing average rate."
    if any(w in r for w in ["smooth", "even", "constant"]):
        return "Leaky Bucket — enforces perfectly smooth output, zero bursts."
    if any(w in r for w in ["simple", "memory", "minimal"]):
        return "Fixed Window — simplest implementation, O(1) memory."
    if any(w in r for w in ["precise", "accurate", "security"]):
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


# ── Agent ─────────────────────────────────────────────────────────────────────

tools            = [search_knowledge_base, get_algorithm_info, recommend_algorithm, calculate_token_bucket]
model            = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Rate Limiting Expert assistant with access to a knowledge base. "
    "For detailed questions, use search_knowledge_base to retrieve accurate information. "
    "Give thorough, well-structured answers using markdown formatting."
))

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_llm(state: AgentState) -> dict:
    response = model_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("llm",   call_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
agent = graph.compile()


def get_response(message: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    return result["messages"][-1].content
