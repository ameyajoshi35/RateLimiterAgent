"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 1 — Talking to an LLM with LangChain                  ║
╚══════════════════════════════════════════════════════════════╝

WHAT IS LANGCHAIN?
  LangChain is a framework that wraps LLM providers (Anthropic, OpenAI, etc.)
  behind a single, consistent interface. You can swap providers without
  changing the rest of your code.

WHAT IS A CHAT MODEL?
  A chat model takes a LIST OF MESSAGES and returns one AI message.
  Messages have roles:
    - SystemMessage  → sets context / persona for the assistant
    - HumanMessage   → what the user says
    - AIMessage      → what the model replies

HOW TO RUN:
  export ANTHROPIC_API_KEY=your_key_here
  uv run step1_llm_basics.py
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── 1. Create a Chat Model ─────────────────────────────────────────────────────
#
# ChatGroq wraps the Groq API. The model= parameter picks which model to use.
# You could swap this for ChatAnthropic("claude-...") or ChatOpenAI("gpt-4o")
# and the rest of the code stays the same — that's the point of LangChain.
#
model = ChatGroq(model="llama-3.3-70b-versatile")

print("=" * 60)
print("STEP 1: Talking to an LLM")
print("=" * 60)


# ── 2. Send a single message ───────────────────────────────────────────────────
#
# .invoke() is the standard method to call ANY LangChain component.
# It takes input → returns output. Here: list of messages → AIMessage.
#
response = model.invoke([
    HumanMessage(content="What is a rate limiter in one sentence?")
])

print(f"\n[Response type]  {type(response).__name__}")   # AIMessage
print(f"[Response content]\n{response.content}")


# ── 3. Add a SystemMessage ─────────────────────────────────────────────────────
#
# SystemMessage is always first. It tells the model who it is and how to behave.
# The model never "says" the system message — it just uses it as instructions.
#
messages = [
    SystemMessage(content="You are an expert in distributed systems. Be concise."),
    HumanMessage(content="Name the 4 main rate limiting algorithms."),
]

response = model.invoke(messages)
print(f"\n[With system prompt]\n{response.content}")


# ── 4. The model is stateless ─────────────────────────────────────────────────
#
# The model has NO memory between calls. Each .invoke() is independent.
# To have a conversation, you must pass ALL previous messages every time.
# LangGraph (step 4) solves this by managing message history for you.
#
follow_up = [
    SystemMessage(content="You are a distributed systems expert."),
    HumanMessage(content="Name the 4 main rate limiting algorithms."),
    response,   # ← the AI's previous reply
    HumanMessage(content="Which one is best for APIs that allow short bursts?"),
]

response2 = model.invoke(follow_up)
print(f"\n[Follow-up (manual history)]\n{response2.content}")

print("\n✓ Step 1 complete — you can now send messages to an LLM.")
