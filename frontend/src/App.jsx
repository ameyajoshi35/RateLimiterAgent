import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const BACKEND_URL = import.meta.env.VITE_API_URL ?? ''
const CHAR_DELAY  = 18   // ms per character — raise to slow down typing

const WELCOME = `Hi! I'm your **Rate Limiting Expert**. I can help you with:

- How rate limiting algorithms work (Token Bucket, Fixed Window, Sliding Window, Leaky Bucket)
- Choosing the right algorithm for your use case
- Distributed rate limiting with Redis
- API best practices and HTTP headers

What would you like to know?`

const SUGGESTED = [
  "How does token bucket work?",
  "What algorithm for bursty API traffic?",
  "How do I rate limit across multiple servers?",
  "What HTTP headers should I return?",
]

const TOOL_META = {
  search_knowledge_base:  { icon: '🔍', sub: 'FAISS vector search · HuggingFace embeddings' },
  get_algorithm_info:     { icon: '📖', sub: '@tool decorator · returns algorithm description' },
  recommend_algorithm:    { icon: '💡', sub: '@tool decorator · matches use-case to algorithm' },
  calculate_token_bucket: { icon: '🧮', sub: '@tool decorator · computes bucket parameters' },
}


// ── Pipeline node components ───────────────────────────────────────────────

function Badge({ variant }) {
  return <span className={`pv-badge pv-badge-${variant}`}>{variant}</span>
}

function Spinner() {
  return <span className="pv-spin" />
}

function Check() {
  return <span className="pv-check">✓</span>
}

function PipelineNode({ node }) {
  switch (node.type) {

    case 'graph_start':
      return (
        <div className="pv-node pv-green">
          <span className="pv-nicon">🚀</span>
          <div className="pv-nbody">
            <div className="pv-ntitle">StateGraph Initialized <Badge variant="langgraph" /></div>
            <div className="pv-nsub">StateGraph.compile() · add_messages reducer · messages state</div>
          </div>
        </div>
      )

    case 'llm':
      return (
        <div className={`pv-node ${node.status === 'active' ? 'pv-blue' : 'pv-done'}`}>
          <div className="pv-nstatus">{node.status === 'active' ? <Spinner /> : <Check />}</div>
          <span className="pv-nicon">🧠</span>
          <div className="pv-nbody">
            <div className="pv-ntitle">LLM Node — Call #{node.call} <Badge variant="langchain" /></div>
            <div className="pv-nsub">ChatGroq(llama-4-scout-17b) · bind_tools(4 tools)</div>
            {node.status === 'active' && (
              <div className="pv-ndetail pv-dim">Invoking LLM with current messages state…</div>
            )}
            {node.status === 'done' && node.decision === 'tools' && (
              <div className="pv-ndetail pv-orange">
                AIMessage has tool_calls → selected: <strong>{node.toolNames.join(', ')}</strong>
              </div>
            )}
            {node.status === 'done' && node.decision === 'answer' && (
              <div className="pv-ndetail pv-green-text">
                AIMessage has no tool_calls → generating final answer
              </div>
            )}
          </div>
        </div>
      )

    case 'routing':
      return (
        <div className="pv-node pv-routing">
          <span className="pv-nicon pv-diamond">◆</span>
          <div className="pv-nbody">
            <div className="pv-ntitle">
              Conditional Edge → <strong>{node.decision === 'tools' ? 'tools node' : 'END'}</strong>
              <Badge variant="langgraph" />
            </div>
            <div className="pv-nsub">add_conditional_edges · tools_condition(state)</div>
            <div className={`pv-ndetail ${node.decision === 'tools' ? 'pv-orange' : 'pv-green-text'}`}>
              {node.decision === 'tools'
                ? 'last message has tool_calls → route to tools'
                : 'last message has no tool_calls → route to END'}
            </div>
          </div>
        </div>
      )

    case 'tool': {
      const meta = TOOL_META[node.tool] || { icon: '🔧', sub: '@tool decorator' }
      return (
        <div className={`pv-node ${node.status === 'active' ? 'pv-amber' : 'pv-done'}`}>
          <div className="pv-nstatus">{node.status === 'active' ? <Spinner /> : <Check />}</div>
          <span className="pv-nicon">{meta.icon}</span>
          <div className="pv-nbody">
            <div className="pv-ntitle">ToolNode: {node.tool} <Badge variant="langchain" /></div>
            <div className="pv-nsub">{meta.sub}</div>
            {Object.entries(node.args || {}).map(([k, v]) => (
              <div key={k} className="pv-ndetail pv-dim">
                <span className="pv-argkey">{k}:</span> {v}
              </div>
            ))}
            {node.status === 'done' && node.preview && (
              <div className="pv-ndetail pv-green-text">→ {node.preview}</div>
            )}
            {node.status === 'done' && (
              <div className="pv-ndetail pv-dim">ToolMessage appended to messages state</div>
            )}
          </div>
        </div>
      )
    }

    case 'graph_end':
      return (
        <div className="pv-node pv-green">
          <span className="pv-nicon">🏁</span>
          <div className="pv-nbody">
            <div className="pv-ntitle">Graph END <Badge variant="langgraph" /></div>
            <div className="pv-nsub">No more tool_calls · graph terminates · returns messages[-1]</div>
          </div>
        </div>
      )

    default:
      return null
  }
}

function PipelineView({ pipeline, streaming }) {
  const [open, setOpen] = useState(true)
  if (!pipeline.length) return null

  return (
    <div className="pv-wrap">
      <button className="pv-header" onClick={() => setOpen(o => !o)}>
        <span className="pv-header-left">
          {streaming
            ? <><span className="pv-spin pv-spin-sm" /> Running ReAct loop…</>
            : '⬡  Agent loop complete'}
        </span>
        <span className="pv-toggle">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="pv-flow">
          {pipeline.map((node, i) => (
            <div key={node.key}>
              {i > 0 && <div className="pv-connector"><span className="pv-line" />↓</div>}
              <PipelineNode node={node} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState([
    { id: 0, role: 'assistant', content: WELCOME, pipeline: [], pending: false, streaming: false },
  ])
  const [input, setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef(null)
  const textareaRef           = useRef(null)
  const tokenQueue            = useRef([])
  const tickerRef             = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // ── Typewriter helpers ─────────────────────────────────────────────────

  const startTicker = (id) => {
    if (tickerRef.current) return
    tickerRef.current = setInterval(() => {
      if (!tokenQueue.current.length) return
      const ch = tokenQueue.current.shift()
      setMessages(prev => prev.map(m =>
        m.id === id ? { ...m, content: (m.content || '') + ch, pending: true } : m
      ))
    }, CHAR_DELAY)
  }

  const stopTicker = (id) => {
    const rem = tokenQueue.current.splice(0)
    if (rem.length) {
      setMessages(prev => prev.map(m =>
        m.id === id ? { ...m, content: (m.content || '') + rem.join('') } : m
      ))
    }
    clearInterval(tickerRef.current)
    tickerRef.current = null
  }

  // ── Pipeline helpers ───────────────────────────────────────────────────

  const pushNode = (id, node) =>
    setMessages(prev => prev.map(m =>
      m.id !== id ? m : { ...m, pipeline: [...m.pipeline, node] }
    ))

  const patchLastNode = (id, pred, patch) =>
    setMessages(prev => prev.map(m => {
      if (m.id !== id) return m
      const p = [...m.pipeline]
      const i = p.findLastIndex(pred)
      if (i >= 0) p[i] = { ...p[i], ...patch }
      return { ...m, pipeline: p }
    }))

  // ── Send ───────────────────────────────────────────────────────────────

  const send = async (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return

    setMessages(prev => [...prev,
      { id: Date.now(), role: 'user', content: msg, pipeline: [], pending: false, streaming: false },
    ])
    setInput('')
    setLoading(true)
    tokenQueue.current = []

    const aid = Date.now() + 1
    setMessages(prev => [...prev,
      { id: aid, role: 'assistant', content: '', pipeline: [], pending: true, streaming: true },
    ])

    const patch = (p) => setMessages(prev => prev.map(m => m.id === aid ? { ...m, ...p } : m))

    let seq = 0
    const key = () => `n${++seq}`

    try {
      const res = await fetch(`${BACKEND_URL}/chat/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: msg }),
      })

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buf     = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop()

        for (const part of parts) {
          const line = part.trim()
          if (!line || line === 'data: [DONE]') continue
          if (!line.startsWith('data: ')) continue
          let ev
          try { ev = JSON.parse(line.slice(6)) } catch { continue }

          if (ev.type === 'pipeline') {
            switch (ev.phase) {
              case 'graph_start':
                pushNode(aid, { key: key(), type: 'graph_start', status: 'done' })
                break
              case 'llm_start':
                pushNode(aid, { key: key(), type: 'llm', status: 'active', call: ev.call, decision: null, toolNames: [] })
                break
              case 'llm_end':
                patchLastNode(aid, n => n.type === 'llm' && n.status === 'active',
                  { status: 'done', decision: ev.decision, toolNames: ev.tool_names || [] })
                pushNode(aid, { key: key(), type: 'routing', status: 'done', decision: ev.decision, toolNames: ev.tool_names || [] })
                break
              case 'tool_start':
                pushNode(aid, { key: key(), type: 'tool', status: 'active', tool: ev.tool, args: ev.args || {}, preview: null })
                break
              case 'tool_end':
                patchLastNode(aid, n => n.type === 'tool' && n.tool === ev.tool && n.status === 'active',
                  { status: 'done', preview: ev.preview })
                break
              case 'graph_end':
                pushNode(aid, { key: key(), type: 'graph_end', status: 'done' })
                patch({ streaming: false })
                break
            }
          } else if (ev.type === 'token') {
            tokenQueue.current.push(...ev.content.split(''))
            startTicker(aid)
          } else if (ev.type === 'error') {
            stopTicker(aid)
            patch({ content: `**Error:** ${ev.message}`, pending: false, streaming: false })
          }
        }
      }

      // wait for typewriter queue to drain, then mark done
      const waitDrain = () => {
        if (!tokenQueue.current.length) {
          clearInterval(tickerRef.current)
          tickerRef.current = null
          setMessages(prev => prev.map(m => m.id === aid ? { ...m, pending: false } : m))
          setLoading(false)
          textareaRef.current?.focus()
        } else {
          setTimeout(waitDrain, CHAR_DELAY * 2)
        }
      }
      waitDrain()
      return  // skip finally's setLoading

    } catch {
      stopTicker(aid)
      patch({ content: '**Error:** Could not reach the backend. Make sure it is running on port 8000.', pending: false, streaming: false })
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const showSuggestions = messages.length === 1

  return (
    <div className="app">

      <header>
        <div className="header-logo">⚡</div>
        <div className="header-text">
          <span className="header-title">Rate Limiter Agent</span>
          <span className="header-sub">LangGraph · RAG · Groq</span>
        </div>
      </header>

      <main>
        {messages.map((msg) => (
          <div key={msg.id} className={`row ${msg.role}`}>
            <div className="avatar">{msg.role === 'assistant' ? '🤖' : '👤'}</div>
            <div className="bubble-wrap">

              {msg.role === 'assistant' && msg.pipeline?.length > 0 && (
                <PipelineView pipeline={msg.pipeline} streaming={msg.streaming} />
              )}

              {msg.pending && !msg.content
                ? <div className="bubble typing"><span /><span /><span /></div>
                : msg.content
                  ? <div className={`bubble${msg.pending ? ' streaming' : ''}`}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                      {msg.pending && <span className="cursor" />}
                    </div>
                  : null
              }

            </div>
          </div>
        ))}

        {showSuggestions && (
          <div className="suggestions">
            {SUGGESTED.map(q => (
              <button key={q} className="suggestion-chip" onClick={() => send(q)}>{q}</button>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about rate limiting… (Enter to send)"
          rows={1}
          disabled={loading}
        />
        <button onClick={() => send()} disabled={loading || !input.trim()}>
          {loading ? '…' : '↑'}
        </button>
      </footer>

    </div>
  )
}
