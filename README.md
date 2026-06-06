# RateLimiter Agent

A step-by-step LangChain + LangGraph agent that answers questions about rate limiting algorithms. Built as a learning project — each file introduces one concept before combining them all in a full ReAct agent.

## Setup

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv), a free [Groq API key](https://console.groq.com)

```bash
# Install dependencies
uv venv .venv --python 3.12
uv pip install -r requirements.txt

# Set your API key
export GROQ_API_KEY=your_key_here
```

## Run

```bash
.venv/bin/python step6_rag_agent.py    # full RAG agent (recommended)
.venv/bin/python step5_full_agent.py   # agent without RAG
```

Or run each step individually to follow the learning path:

```bash
.venv/bin/python step1_llm_basics.py
.venv/bin/python step2_prompts_and_chains.py
.venv/bin/python step3_tools.py
.venv/bin/python step4_langgraph_intro.py
.venv/bin/python step5_full_agent.py
.venv/bin/python step6_rag_agent.py
```

> **Note:** Step 6 downloads the embedding model (~90MB) on first run. It's cached after that.

## Learning Path

| File | Concept |
|---|---|
| `step1_llm_basics.py` | Chat models, messages, `.invoke()`, statelessness |
| `step2_prompts_and_chains.py` | Prompt templates, LCEL `\|` pipe, `.stream()`, `.batch()` |
| `step3_tools.py` | `@tool` decorator, `bind_tools()`, tool call cycle |
| `step4_langgraph_intro.py` | State, nodes, edges, conditional routing |
| `step5_full_agent.py` | Full ReAct loop with LangGraph |
| `step6_rag_agent.py` | RAG — embeddings, vector store, semantic search |

## How the Agent Works

### ReAct Loop (steps 5 & 6)

The agent uses the **ReAct** (Reasoning + Acting) pattern — it loops between thinking and calling tools until it has enough information to answer.

```
START → [llm] → has tool_calls? → YES → [tools] → back to [llm]
                                → NO  → END
```

### RAG Pipeline (step 6)

RAG (Retrieval-Augmented Generation) lets the agent search a knowledge base instead of relying only on the LLM's training data.

```
INDEXING (once at startup)
  documents → split into chunks → embed with all-MiniLM-L6-v2 → FAISS index

RETRIEVAL (at query time)
  question → embed → find 3 nearest chunks → pass as context to LLM
```

### Tools

| Tool | Available in | Description |
|---|---|---|
| `get_algorithm_info` | Steps 3, 5, 6 | Brief description of a rate limiting algorithm |
| `recommend_algorithm` | Steps 3, 5, 6 | Recommends an algorithm based on requirements |
| `calculate_token_bucket` | Steps 3, 5, 6 | Calculates Token Bucket config for a given rate and burst size |
| `search_knowledge_base` | Step 6 | Semantic search over the rate limiting knowledge base |

### Example (step 6 RAG agent)

```
User: What HTTP headers should I return when rate limiting an API?

[llm] → search_knowledge_base("HTTP headers rate limiting API")
      → retrieves chunk from api_best_practices doc:
        "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset..."
[llm] → final answer with retrieved context
```

## Knowledge Base (step 6)

The RAG agent indexes 6 documents covering:

| Document | Content |
|---|---|
| Token Bucket Guide | Deep dive: burst handling, parameters, best use cases |
| Fixed Window Guide | Boundary spike problem explained with examples |
| Sliding Window Log Guide | Memory trade-offs, precision guarantees |
| Leaky Bucket Guide | Queue mechanics, comparison with Token Bucket |
| Distributed Rate Limiting | Redis strategies, sticky sessions, approximate counting |
| API Best Practices | HTTP headers, HTTP 429, tiered limits, graceful degradation |

## Models

| Step | Model | Reason |
|---|---|---|
| Steps 1–2, 4 | `llama-3.3-70b-versatile` | Fast, no tool calling needed |
| Steps 3, 5, 6 | `meta-llama/llama-4-scout-17b-16e-instruct` | Reliable tool calling |
