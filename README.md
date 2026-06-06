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
.venv/bin/python step5_full_agent.py   # full agent
```

Or run each step individually to follow the learning path:

```bash
.venv/bin/python step1_llm_basics.py
.venv/bin/python step2_prompts_and_chains.py
.venv/bin/python step3_tools.py
.venv/bin/python step4_langgraph_intro.py
.venv/bin/python step5_full_agent.py
```

## Learning Path

| File | Concept |
|---|---|
| `step1_llm_basics.py` | Chat models, messages, `.invoke()`, statelessness |
| `step2_prompts_and_chains.py` | Prompt templates, LCEL `\|` pipe, `.stream()`, `.batch()` |
| `step3_tools.py` | `@tool` decorator, `bind_tools()`, tool call cycle |
| `step4_langgraph_intro.py` | State, nodes, edges, conditional routing |
| `step5_full_agent.py` | Full ReAct loop with LangGraph |

## How the Agent Works

The agent uses the **ReAct** (Reasoning + Acting) pattern — it loops between thinking and calling tools until it has enough information to answer.

```
START → [llm] → has tool_calls? → YES → [tools] → back to [llm]
                                → NO  → END
```

### Tools

| Tool | Description |
|---|---|
| `get_algorithm_info` | Returns a description of a rate limiting algorithm |
| `recommend_algorithm` | Recommends an algorithm based on requirements |
| `calculate_token_bucket` | Calculates Token Bucket config for given rate and burst size |

### Example

```
User: I need to rate limit an API with bursty traffic. What algorithm should I use?

[llm] → calls recommend_algorithm("bursty traffic")
[llm] → calls get_algorithm_info("token_bucket")
[llm] → final answer: "Use Token Bucket — it tolerates short bursts while enforcing an average rate..."
```

## Models

| Step | Model | Reason |
|---|---|---|
| Steps 1–2, 4 | `llama-3.3-70b-versatile` | Fast, no tool calling needed |
| Steps 3, 5 | `meta-llama/llama-4-scout-17b-16e-instruct` | Reliable tool calling |
