"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 3 — Tools                                             ║
╚══════════════════════════════════════════════════════════════╝

WHAT ARE TOOLS?
  Tools are Python functions the LLM can choose to call.
  This is what separates a "chatbot" from an "agent":
    - Chatbot: only generates text
    - Agent:   can take ACTIONS (look things up, calculate, call APIs…)

HOW DOES TOOL CALLING WORK?
  1. You bind tools to the model: model.bind_tools([tool1, tool2])
  2. You send a message to the model
  3. The model may return an AIMessage with .tool_calls (instead of text)
     Each tool_call has: name, args
  4. YOUR CODE runs the tool with those args and gets a result
  5. You send that result back as a ToolMessage
  6. The model reads the result and continues

  The model never runs tools itself — it only REQUESTS them.
  YOU are responsible for executing them and returning results.

HOW TO DEFINE A TOOL?
  Use the @tool decorator. Two things matter:
    - The docstring  → the LLM reads this to decide WHEN to call it
    - Type hints     → the LLM uses these to format its arguments correctly

HOW TO RUN:
  export ANTHROPIC_API_KEY=your_key_here
  uv run step3_tools.py
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

print("=" * 60)
print("STEP 3: Tools")
print("=" * 60)


# ── 1. Define tools with @tool ────────────────────────────────────────────────
#
# The @tool decorator wraps a function into a LangChain Tool object.
# The docstring is critical — it's the description the LLM sees.
#
@tool
def get_algorithm_info(algorithm: str) -> str:
    """Returns a technical description of a rate limiting algorithm.
    Use this when the user asks how a specific algorithm works.
    Valid values: token_bucket, fixed_window, sliding_window_log, leaky_bucket
    """
    info = {
        "token_bucket": (
            "Tokens accumulate at a fixed rate up to a capacity cap. "
            "Each request consumes one token. Burst traffic is absorbed "
            "as long as the bucket isn't empty."
        ),
        "fixed_window": (
            "Time is split into fixed slots (e.g., per second). A counter "
            "resets each slot. Simple, but allows 2× the limit in requests "
            "straddling two consecutive windows."
        ),
        "sliding_window_log": (
            "Every allowed request's timestamp is stored. On each call, "
            "expired timestamps are evicted and the window is enforced "
            "precisely. Uses O(limit) memory."
        ),
        "leaky_bucket": (
            "Requests enter a fixed-size queue and drain at a constant rate. "
            "Output is perfectly smooth — no bursts ever pass through. "
            "Excess requests are rejected when the queue is full."
        ),
    }
    key = algorithm.lower().replace(" ", "_")
    return info.get(key, f"Unknown algorithm '{algorithm}'. "
                         f"Try: token_bucket, fixed_window, sliding_window_log, leaky_bucket")


@tool
def recommend_algorithm(requirements: str) -> str:
    """Recommends a rate limiting algorithm based on the user's requirements.
    Use this when the user describes what they need (e.g. burst tolerance,
    smooth output, simplicity, precision) and wants a recommendation.
    """
    r = requirements.lower()
    if any(w in r for w in ["burst", "spike", "peak"]):
        return "Token Bucket — tolerates short bursts while enforcing an average rate."
    if any(w in r for w in ["smooth", "even", "constant", "steady"]):
        return "Leaky Bucket — enforces perfectly smooth output, zero bursts."
    if any(w in r for w in ["simple", "low memory", "minimal"]):
        return "Fixed Window — simplest implementation, O(1) memory."
    if any(w in r for w in ["precise", "accurate", "exact"]):
        return "Sliding Window Log — most precise, no boundary effects."
    return "Token Bucket — solid default: simple, burst-tolerant, O(1) memory."


@tool
def calculate_token_bucket(requests_per_second: float, burst_size: int) -> str:
    """Calculates Token Bucket parameters given a desired rate and burst size.
    Use this when the user provides numeric requirements like requests per second.
    Args:
        requests_per_second: the desired average throughput
        burst_size: max number of requests allowed in an instant
    """
    refill_rate = requests_per_second
    capacity    = burst_size
    refill_ms   = round(1000 / refill_rate, 2) if refill_rate > 0 else float("inf")
    return (
        f"Token Bucket config:\n"
        f"  capacity    = {capacity} tokens  (max burst)\n"
        f"  refill rate = {refill_rate} tokens/second\n"
        f"  refill interval = {refill_ms} ms per token"
    )


# ── 2. Inspect what the LLM sees ──────────────────────────────────────────────
tools = [get_algorithm_info, recommend_algorithm, calculate_token_bucket]

print("\n[Tool metadata the LLM reads]")
for t in tools:
    print(f"\n  name: {t.name}")
    print(f"  description: {t.description[:80]}...")
    print(f"  args schema: {t.args}")


# ── 3. Bind tools to the model ────────────────────────────────────────────────
#
# bind_tools() tells the model which tools are available.
# The model can now choose to return tool_calls instead of plain text.
#
model_with_tools = model.bind_tools(tools)


# ── 4. The model requests a tool call ─────────────────────────────────────────
print("\n[Sending: 'How does token bucket work?']")
response = model_with_tools.invoke([
    HumanMessage(content="How does the token bucket algorithm work?")
])

print(f"  Has text:       {bool(response.content)}")
print(f"  Has tool_calls: {bool(response.tool_calls)}")
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"  → tool: {tc['name']}, args: {tc['args']}")


# ── 5. Execute the tool and return the result ─────────────────────────────────
#
# We manually execute the tool here. In step 5, LangGraph does this for us.
#
tool_map = {t.name: t for t in tools}

messages = [HumanMessage(content="How does the token bucket algorithm work?"), response]

for tool_call in response.tool_calls:
    tool_fn     = tool_map[tool_call["name"]]
    tool_result = tool_fn.invoke(tool_call["args"])

    # ToolMessage carries the result back. The id links it to the request.
    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    print(f"\n[Tool result for '{tool_call['name']}']\n  {tool_result}")


# ── 6. Model reads the result and gives a final answer ───────────────────────
final = model_with_tools.invoke(messages)
print(f"\n[Final answer from model]\n{final.content}")

print("\n✓ Step 3 complete — you can define tools and run a single tool-call cycle.")
