"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 2 — Prompts & Chains (LCEL)                           ║
╚══════════════════════════════════════════════════════════════╝

WHAT IS A PROMPT TEMPLATE?
  Instead of hardcoding message text, use {variables} as placeholders.
  This lets you reuse the same prompt shape with different inputs.

WHAT IS LCEL (LangChain Expression Language)?
  LCEL uses the | (pipe) operator to chain components together, just
  like Unix pipes. Each component's output becomes the next one's input.

  prompt | model | output_parser
    ↑         ↑         ↑
  fills     generates  strips AIMessage
  variables  text       → plain string

  This is called a "Runnable chain". You can .invoke(), .stream(),
  or .batch() any chain exactly like a single component.

HOW TO RUN:
  export ANTHROPIC_API_KEY=your_key_here
  uv run step2_prompts_and_chains.py
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGroq(model="llama-3.3-70b-versatile")

print("=" * 60)
print("STEP 2: Prompts and Chains")
print("=" * 60)


# ── 1. ChatPromptTemplate ──────────────────────────────────────────────────────
#
# .from_messages() takes a list of (role, template_string) tuples.
# Anything in {curly_braces} is a variable you fill in at call time.
#
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a rate limiting expert. Keep answers under 3 sentences."),
    ("human", "Explain the {algorithm} algorithm. Focus on: {focus}."),
])

# You can inspect the rendered prompt without calling the model:
rendered = prompt.invoke({"algorithm": "Token Bucket", "focus": "burst handling"})
print("\n[Rendered prompt messages]")
for msg in rendered.messages:
    print(f"  [{msg.type:6}] {msg.content}")


# ── 2. StrOutputParser ────────────────────────────────────────────────────────
#
# model.invoke() returns an AIMessage object.
# StrOutputParser extracts just the .content string from it.
# Without this, your chain output is an AIMessage, not a plain string.
#
parser = StrOutputParser()


# ── 3. Build a chain with | ───────────────────────────────────────────────────
#
# The pipe operator connects Runnables. Under the hood:
#   chain.invoke(input)
#     → prompt.invoke(input)         fills variables → ChatPromptValue
#     → model.invoke(prompt_value)   calls API       → AIMessage
#     → parser.invoke(ai_message)    extracts text   → str
#
chain = prompt | model | parser

result = chain.invoke({"algorithm": "Token Bucket", "focus": "burst handling"})
print(f"\n[Chain output — plain string]\n{result}")


# ── 4. Reuse the chain with different inputs ──────────────────────────────────
#
# The whole point of templates: same chain, different data.
#
algorithms = [
    {"algorithm": "Fixed Window",      "focus": "boundary spikes"},
    {"algorithm": "Sliding Window Log","focus": "memory usage"},
    {"algorithm": "Leaky Bucket",      "focus": "output smoothness"},
]

print("\n[Batch: all algorithms]")
results = chain.batch(algorithms)   # runs all 3 in parallel
for inp, out in zip(algorithms, results):
    print(f"\n  {inp['algorithm']}:\n  {out}")


# ── 5. Streaming ──────────────────────────────────────────────────────────────
#
# .stream() yields partial text chunks as they arrive from the API.
# Useful for UIs where you want to show text appearing word by word.
#
print("\n[Streaming Token Bucket explanation]")
for chunk in chain.stream({"algorithm": "Token Bucket", "focus": "refill rate"}):
    print(chunk, end="", flush=True)
print()

print("\n✓ Step 2 complete — you can now build reusable, composable chains.")
