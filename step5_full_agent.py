"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 5 — Full ReAct Agent with LangGraph                   ║
╚══════════════════════════════════════════════════════════════╝

WHAT IS A ReAct AGENT?
  ReAct = Reasoning + Acting
  The agent alternates between:
    - Reasoning: the LLM thinks about what to do
    - Acting:    the LLM calls a tool (and we execute it)
  ...until the LLM decides it has enough information to answer.

THE AGENT LOOP (as a graph):

    ┌─────────────────────────────┐
    │                             │
    ▼                             │
  [llm] ──── has tool_calls? ── YES ──→ [tools]
    │
    NO
    │
    ▼
  [END]

  - "llm" node:   calls the model with full message history
  - "tools" node: executes every tool_call the model requested
  - Conditional edge: if the model's last message has tool_calls → go to
    "tools", otherwise → END

WHAT IS ToolNode?
  LangGraph's built-in ToolNode handles step 4 from step 3 automatically:
    - Finds which tool to call
    - Executes it
    - Wraps the result in a ToolMessage
    - Returns updated messages
  You don't have to write that loop yourself.

HOW TO RUN:
  export ANTHROPIC_API_KEY=your_key_here
  uv run step5_full_agent.py
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

print("=" * 60)
print("STEP 5: Full ReAct Agent")
print("=" * 60)


# ── 1. Tools (same as step 3) ─────────────────────────────────────────────────

@tool
def get_algorithm_info(algorithm: str) -> str:
    """Returns a technical description of a rate limiting algorithm.
    Use this when the user asks how a specific algorithm works.
    Valid values: token_bucket, fixed_window, sliding_window_log, leaky_bucket
    """
    info = {
        "token_bucket": (
            "Tokens accumulate at a fixed rate up to a capacity cap. "
            "Each request consumes one token. Allows short bursts."
        ),
        "fixed_window": (
            "Time is split into fixed slots. A counter resets each slot. "
            "Allows 2× the limit at window boundaries."
        ),
        "sliding_window_log": (
            "Every request's timestamp is logged. Expired entries are evicted. "
            "Precise, no boundary spikes. O(limit) memory."
        ),
        "leaky_bucket": (
            "Requests queue up and drain at a constant rate. "
            "Perfectly smooth output, no bursts pass through."
        ),
    }
    key = algorithm.lower().replace(" ", "_")
    return info.get(key, f"Unknown algorithm '{algorithm}'.")


@tool
def recommend_algorithm(requirements: str) -> str:
    """Recommends a rate limiting algorithm based on described requirements.
    Use this when the user describes what they need and wants a recommendation.
    """
    r = requirements.lower()
    if any(w in r for w in ["burst", "spike", "peak"]):
        return "Token Bucket — tolerates short bursts while enforcing average rate."
    if any(w in r for w in ["smooth", "even", "constant"]):
        return "Leaky Bucket — enforces perfectly smooth output, zero bursts."
    if any(w in r for w in ["simple", "memory", "minimal"]):
        return "Fixed Window — simplest implementation, O(1) memory."
    if any(w in r for w in ["precise", "accurate", "exact"]):
        return "Sliding Window Log — most precise, no boundary effects."
    return "Token Bucket — solid default: simple, burst-tolerant, O(1) memory."


@tool
def calculate_token_bucket(requests_per_second: str, burst_size: str) -> str:
    """Calculates Token Bucket parameters for a given rate and burst size.
    Use this when the user provides numeric throughput requirements.
    requests_per_second: the desired average throughput (e.g. '100')
    burst_size: max number of requests allowed in an instant (e.g. '200')
    """
    requests_per_second = float(requests_per_second)
    burst_size = int(burst_size)
    refill_ms = round(1000 / requests_per_second, 2) if requests_per_second > 0 else 0
    return (
        f"Token Bucket config:\n"
        f"  capacity         = {burst_size} tokens\n"
        f"  refill rate      = {requests_per_second} tokens/sec\n"
        f"  refill interval  = {refill_ms} ms per token"
    )


tools = [get_algorithm_info, recommend_algorithm, calculate_token_bucket]


# ── 2. State ──────────────────────────────────────────────────────────────────
#
# Same pattern as step 4. add_messages accumulates the conversation.
#
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── 3. Model with tools bound ─────────────────────────────────────────────────
#
# bind_tools() gives the model the tool schemas so it knows what it can call.
#
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
model_with_tools = model.bind_tools(tools)


# ── 4. Nodes ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Rate Limiter Expert assistant. "
    "Use the available tools to answer questions about rate limiting algorithms. "
    "Always use a tool to look up information before answering."
))

def call_llm(state: AgentState) -> dict:
    """
    Node: sends the full message history to the LLM.
    The model either:
      a) Returns a plain text reply  → we're done
      b) Returns tool_calls          → we need to execute them
    """
    # Prepend the system message (it's not stored in state, just injected here)
    all_messages = [SYSTEM_PROMPT] + state["messages"]
    response = model_with_tools.invoke(all_messages)

    if response.tool_calls:
        print(f"  [llm] → requesting {len(response.tool_calls)} tool call(s):")
        for tc in response.tool_calls:
            print(f"         {tc['name']}({tc['args']})")
    else:
        print(f"  [llm] → final answer ready")

    return {"messages": [response]}


# ToolNode is a pre-built LangGraph node. It:
#   1. Reads the last AIMessage's tool_calls
#   2. Calls the matching tool function
#   3. Wraps results in ToolMessages and returns them
tool_node = ToolNode(tools)


# ── 5. Conditional edge function ──────────────────────────────────────────────
#
# After "llm" runs, we check: did the model ask for tools?
# Return "tools" to loop back, or END to stop.
#
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # → run tools, then come back to llm
    return END          # → no more tool calls, we're done


# ── 6. Build the graph ────────────────────────────────────────────────────────

graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm",   call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")

# Conditional edge: after "llm", call should_continue() to decide where to go
graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",  # if should_continue returns "tools" → go to tools node
        END: END,          # if should_continue returns END     → stop
    }
)

# After tools run, always go back to the LLM (the loop)
graph_builder.add_edge("tools", "llm")

agent = graph_builder.compile()


# ── 7. Run the agent ──────────────────────────────────────────────────────────

def chat(question: str):
    print(f"\n{'─'*60}")
    print(f"User: {question}")
    print(f"{'─'*60}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # The last message is the model's final plain-text answer
    final = result["messages"][-1]
    print(f"\nAssistant: {final.content}")
    return result


# Test with questions that require different tools and multiple steps
chat("How does the token bucket algorithm work?")

chat("I need to rate limit an API that receives bursty traffic. What algorithm should I use, and can you explain how it works?")

chat("I want to allow 100 requests per second with bursts up to 200. What should my token bucket config look like?")

print("\n✓ Step 5 complete — you have a fully working ReAct agent!")
print("\n  The loop was:")
print("  START → llm → (tool_calls?) → tools → llm → ... → END")
