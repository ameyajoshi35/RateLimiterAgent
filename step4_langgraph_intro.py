"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 4 — LangGraph Basics                                  ║
╚══════════════════════════════════════════════════════════════╝

WHAT IS LANGGRAPH?
  LangGraph lets you model your application as a directed graph:
    - Nodes:  functions that do work (call LLM, run a tool, transform data)
    - Edges:  arrows between nodes (can be conditional / branching)
    - State:  a typed dictionary shared by all nodes

  Think of it as a flowchart where nodes are boxes, edges are arrows,
  and the State is a shared clipboard every box can read and write.

WHY LANGGRAPH INSTEAD OF A PLAIN LCEL CHAIN?
  LCEL chains are LINEAR: A → B → C. They can't loop.
  Agents need LOOPS: call LLM → run tool → call LLM again → ...
  LangGraph supports any graph shape: loops, branches, parallel paths.

KEY CONCEPTS:
  StateGraph(State)   — creates a graph with a specific state shape
  add_node(name, fn)  — registers a function as a node
  add_edge(a, b)      — always go from a to b
  add_conditional_edges(a, fn) — decide where to go based on state
  graph.compile()     — locks the graph and returns a runnable
  START               — the virtual entry node
  END                 — the virtual exit node

  add_messages reducer — instead of REPLACING the messages list,
                         it APPENDS new messages. This is how history
                         accumulates across nodes.

HOW TO RUN:
  export ANTHROPIC_API_KEY=your_key_here
  uv run step4_langgraph_intro.py
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

print("=" * 60)
print("STEP 4: LangGraph Basics")
print("=" * 60)


# ── 1. Define State ───────────────────────────────────────────────────────────
#
# State is a TypedDict — a plain Python dict with type hints.
# Every node receives the current State and returns a PARTIAL update
# (only the keys it changed).
#
# Annotated[list, add_messages] means:
#   - the field holds a list
#   - when a node returns new messages, ADD them to the list (don't replace it)
#
class State(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int     # plain value — last write wins (no reducer)
    topic: str          # plain value — last write wins


# ── 2. Define Nodes ───────────────────────────────────────────────────────────
#
# A node is any function: (state: State) → dict
# Return only the keys you want to update. Other keys stay unchanged.
#
def extract_topic(state: State) -> dict:
    """Reads the first human message and records the topic."""
    first_msg  = state["messages"][0].content
    topic      = first_msg.split()[-1]          # last word as a rough topic
    step_count = state.get("step_count", 0) + 1

    print(f"  [extract_topic] found topic='{topic}', step={step_count}")
    return {"topic": topic, "step_count": step_count}


def add_context(state: State) -> dict:
    """Appends an AI message with context about the extracted topic."""
    topic      = state["topic"]
    step_count = state.get("step_count", 0) + 1

    context_msg = AIMessage(
        content=f"(Context injected by add_context node: topic is '{topic}')"
    )
    print(f"  [add_context] injecting context for topic='{topic}', step={step_count}")

    # Because of add_messages reducer, returning a list here APPENDS to history.
    return {"messages": [context_msg], "step_count": step_count}


def summarise(state: State) -> dict:
    """Appends a final summary message."""
    step_count = state.get("step_count", 0) + 1
    total_msgs = len(state["messages"])

    summary_msg = AIMessage(
        content=f"Pipeline complete. Processed topic='{state['topic']}' "
                f"in {step_count} steps. Total messages: {total_msgs + 1}."
    )
    print(f"  [summarise] wrapping up, step={step_count}")
    return {"messages": [summary_msg], "step_count": step_count}


# ── 3. Build the Graph ────────────────────────────────────────────────────────
#
# StateGraph(State) — creates a graph that uses our State shape
#
graph_builder = StateGraph(State)

# Register nodes. The string name is used in add_edge() calls.
graph_builder.add_node("extract_topic", extract_topic)
graph_builder.add_node("add_context",   add_context)
graph_builder.add_node("summarise",     summarise)

# Add edges (execution order)
graph_builder.add_edge(START,          "extract_topic")   # start here
graph_builder.add_edge("extract_topic","add_context")
graph_builder.add_edge("add_context",  "summarise")
graph_builder.add_edge("summarise",    END)                # stop here

# compile() validates the graph and returns a Runnable
graph = graph_builder.compile()


# ── 4. Run the graph ──────────────────────────────────────────────────────────
print("\n[Running graph with initial state]")
initial_state = {
    "messages": [HumanMessage(content="Tell me about token_bucket")],
    "step_count": 0,
    "topic": "",
}

final_state = graph.invoke(initial_state)

print("\n[Final state]")
print(f"  topic:      {final_state['topic']}")
print(f"  step_count: {final_state['step_count']}")
print(f"  messages ({len(final_state['messages'])}):")
for msg in final_state["messages"]:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"    [{role}] {msg.content}")


# ── 5. Conditional edges ─────────────────────────────────────────────────────
#
# add_conditional_edges(node, fn, mapping) — instead of always going to the
# same next node, fn(state) returns a STRING key that the mapping resolves
# to a destination node. This is how you implement IF/ELSE in a graph.
#
print("\n" + "=" * 60)
print("Conditional edges example")
print("=" * 60)

class SimpleState(TypedDict):
    value: int

def check_value(state: SimpleState) -> dict:
    print(f"  [check_value] value={state['value']}")
    return {}

def route(state: SimpleState) -> str:
    """This function decides the next node. Returns a node name."""
    return "high" if state["value"] >= 5 else "low"

def high_branch(state: SimpleState) -> dict:
    print("  [high_branch] value is HIGH")
    return {}

def low_branch(state: SimpleState) -> dict:
    print("  [low_branch] value is LOW")
    return {}

g2 = StateGraph(SimpleState)
g2.add_node("check",  check_value)
g2.add_node("high",   high_branch)
g2.add_node("low",    low_branch)

g2.add_edge(START, "check")
# route() returns "high" or "low" → maps to a node name
g2.add_conditional_edges("check", route, {"high": "high", "low": "low"})
g2.add_edge("high", END)
g2.add_edge("low",  END)

g2 = g2.compile()

print("\n[value=3]")
g2.invoke({"value": 3})

print("[value=8]")
g2.invoke({"value": 8})

print("\n✓ Step 4 complete — you understand State, Nodes, Edges, and Conditionals.")
