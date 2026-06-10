import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import agent, get_response

app = FastAPI(title="Rate Limiter Agent API")

# ALLOWED_ORIGINS: comma-separated list of allowed frontend origins
# e.g. "http://localhost:5173,https://my-frontend.onrender.com"
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [o.strip() for o in _origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(response=get_response(request.message))


# ── Streaming endpoint ────────────────────────────────────────────────────────
#
# Pipeline event types sent to the frontend:
#   {"type": "pipeline", "phase": "graph_start"}
#   {"type": "pipeline", "phase": "llm_start",  "call": N}
#   {"type": "pipeline", "phase": "llm_end",    "decision": "tools"|"answer", "tool_names": [...]}
#   {"type": "pipeline", "phase": "tool_start", "tool": "...", "args": {...}}
#   {"type": "pipeline", "phase": "tool_end",   "tool": "...", "preview": "..."}
#   {"type": "pipeline", "phase": "graph_end"}
#   {"type": "token",    "content": "..."}      ← one per LLM output token
#   {"type": "error",    "message": "..."}
#   data: [DONE]
#
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            llm_call_count = 0
            graph_started  = False

            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # ── LLM node starting ─────────────────────────────────────
                if kind == "on_chat_model_start" and node == "llm":
                    if not graph_started:
                        graph_started = True
                        yield _sse({"type": "pipeline", "phase": "graph_start"})
                    llm_call_count += 1
                    yield _sse({"type": "pipeline", "phase": "llm_start", "call": llm_call_count})

                # ── LLM node done — emit routing decision ─────────────────
                elif kind == "on_chat_model_end" and node == "llm":
                    output     = event["data"].get("output")
                    tool_calls = getattr(output, "tool_calls", []) if output else []
                    yield _sse({
                        "type":       "pipeline",
                        "phase":      "llm_end",
                        "decision":   "tools" if tool_calls else "answer",
                        "tool_names": [tc["name"] for tc in tool_calls],
                    })

                # ── Tool starting ─────────────────────────────────────────
                elif kind == "on_tool_start":
                    args  = event["data"].get("input") or {}
                    short = {k: (str(v)[:80] + "…" if len(str(v)) > 80 else str(v)) for k, v in args.items()}
                    yield _sse({"type": "pipeline", "phase": "tool_start",
                                "tool": event["name"], "args": short})

                # ── Tool done ─────────────────────────────────────────────
                elif kind == "on_tool_end":
                    out     = event["data"].get("output", "")
                    content = out.content if hasattr(out, "content") else str(out)
                    yield _sse({"type": "pipeline", "phase": "tool_end",
                                "tool": event["name"], "preview": _preview(event["name"], content)})

                # ── LLM token (final answer only) ─────────────────────────
                elif kind == "on_chat_model_stream" and node == "llm":
                    chunk = event["data"]["chunk"]
                    if chunk.content and not getattr(chunk, "tool_call_chunks", []):
                        yield _sse({"type": "token", "content": chunk.content})

            yield _sse({"type": "pipeline", "phase": "graph_end"})
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def _preview(tool_name: str, content: str) -> str:
    if tool_name == "search_knowledge_base":
        n = content.count("---") + 1
        return f"Retrieved {n} relevant chunk(s) from knowledge base"
    return content[:120] + "…" if len(content) > 120 else content
