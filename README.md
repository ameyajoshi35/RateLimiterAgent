# RateLimiter Agent

A step-by-step LangChain + LangGraph agent that answers questions about rate limiting algorithms. Built as a learning project — each file introduces one concept before combining them all in a full ReAct agent with RAG, token streaming, and a React chat UI that visualizes the agent loop in real time.

## Architecture

```
frontend/   ←  React + Vite chat UI with live agent loop visualization
backend/    ←  FastAPI server wrapping the RAG agent
step*.py    ←  Step-by-step learning files
```

## Setup

**Requirements:** Python 3.12+, Node.js 18+, [uv](https://github.com/astral-sh/uv), a free [Groq API key](https://console.groq.com)

```bash
# Python dependencies
uv venv .venv --python 3.12
uv pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..

# Set your API key
export GROQ_API_KEY=your_key_here
```

## Running the UI

Open two terminals:

**Terminal 1 — Backend:**
```bash
cd backend
GROQ_API_KEY=your_key_here /path/to/.venv/bin/uvicorn main:app --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

> **Note:** The backend downloads the embedding model (~90MB) on first run. It's cached after that.

## Running the CLI (no UI)

```bash
.venv/bin/python step6_rag_agent.py    # full RAG agent
.venv/bin/python step5_full_agent.py   # agent without RAG
```

## Learning Path

Run each file in order to build up from scratch:

| File | Concept |
|---|---|
| `step1_llm_basics.py` | Chat models, messages, `.invoke()`, statelessness |
| `step2_prompts_and_chains.py` | Prompt templates, LCEL `\|` pipe, `.stream()`, `.batch()` |
| `step3_tools.py` | `@tool` decorator, `bind_tools()`, tool call cycle |
| `step4_langgraph_intro.py` | State, nodes, edges, conditional routing |
| `step5_full_agent.py` | Full ReAct loop with LangGraph |
| `step6_rag_agent.py` | RAG — embeddings, vector store, semantic search |

## How It Works

### React UI

The frontend is a chat interface built with React + Vite:
- **Agent loop visualization** — every response shows a live pipeline flowchart of the ReAct loop
- **Token streaming** — answer text types out character by character as the LLM generates it
- **Typewriter effect** — tokens are queued and released at a human-readable pace
- Suggestion chips on first load for quick questions
- `Enter` to send, `Shift+Enter` for a new line

### Agent Loop Visualization

Each response displays a collapsible **Agent Loop** panel that shows every step the agent takes in real time:

```
🚀 StateGraph Initialized          [langgraph]
   StateGraph.compile() · add_messages reducer · messages state
   ↓
🧠 LLM Node — Call #1              [langchain]   ← spins while active
   ChatGroq(llama-4-scout-17b) · bind_tools(4 tools)
   AIMessage has tool_calls → selected: get_algorithm_info
   ↓
◆  Conditional Edge → tools node   [langgraph]
   add_conditional_edges · tools_condition(state)
   last message has tool_calls → route to tools
   ↓
📖 ToolNode: get_algorithm_info    [langchain]   ← spins while active
   @tool decorator · returns algorithm description
   algorithm: token_bucket
   → Tokens refill at a fixed rate up to a capacity cap...
   ↓
🧠 LLM Node — Call #2              [langchain]
   ChatGroq sees ToolMessage in messages state
   AIMessage has no tool_calls → generating final answer
   ↓
◆  Conditional Edge → END          [langgraph]
   no tool_calls → route to END
   ↓
🏁 Graph END                       [langgraph]
   messages[-1].content → response
```

Nodes are color-coded (blue = active, green = done) and each badge identifies which framework feature is responsible.

### SSE Streaming

The backend uses `agent.astream_events(version="v2")` to emit granular events over Server-Sent Events:

| Event | Trigger |
|---|---|
| `pipeline: graph_start` | First LLM call detected |
| `pipeline: llm_start` | `on_chat_model_start` for the `llm` node |
| `pipeline: llm_end` | `on_chat_model_end` — includes routing decision |
| `pipeline: tool_start` | `on_tool_start` — includes tool name and args |
| `pipeline: tool_end` | `on_tool_end` — includes result preview |
| `pipeline: graph_end` | Stream complete |
| `token` | `on_chat_model_stream` — individual LLM output tokens |

### ReAct Agent Loop

```
START → [llm] → has tool_calls? → YES → [tools] → back to [llm]
                                → NO  → END
```

### RAG Pipeline

```
INDEXING (once at startup)
  documents → split into chunks → embed with all-MiniLM-L6-v2 → FAISS index

RETRIEVAL (at query time)
  question → embed → find 3 nearest chunks → pass as context to LLM
```

### Tools

| Tool | Description |
|---|---|
| `search_knowledge_base` | Semantic search over the rate limiting knowledge base |
| `get_algorithm_info` | Brief description of a rate limiting algorithm |
| `recommend_algorithm` | Recommends an algorithm based on requirements |
| `calculate_token_bucket` | Calculates Token Bucket config for a given rate and burst size |

### Example

```
User: What HTTP headers should I return when rate limiting an API?

[llm] → search_knowledge_base("HTTP headers rate limiting")
      → retrieves chunk: "X-RateLimit-Limit, X-RateLimit-Remaining..."
[llm] → final answer using retrieved context
```

## Knowledge Base

The RAG agent indexes 6 documents:

| Document | Content |
|---|---|
| Token Bucket Guide | Burst handling, parameters, best use cases |
| Fixed Window Guide | Boundary spike problem with examples |
| Sliding Window Log Guide | Memory trade-offs, precision guarantees |
| Leaky Bucket Guide | Queue mechanics, comparison with Token Bucket |
| Distributed Rate Limiting | Redis strategies, sticky sessions, approximate counting |
| API Best Practices | HTTP headers, HTTP 429, tiered limits, graceful degradation |

## Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-4-scout-17b` for tool calling, `llama-3.3-70b` for text) |
| Agent framework | LangGraph — `StateGraph`, `ToolNode`, `add_conditional_edges` |
| RAG | LangChain + HuggingFace embeddings (`all-MiniLM-L6-v2`) + FAISS |
| Streaming | `astream_events(version="v2")` → Server-Sent Events |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + Vite + react-markdown |
