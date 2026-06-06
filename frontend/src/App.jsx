import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const BACKEND_URL = 'http://localhost:8000'

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

export default function App() {
  const [messages, setMessages]   = useState([{ role: 'assistant', content: WELCOME }])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const bottomRef                 = useRef(null)
  const textareaRef               = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return

    setMessages(prev => [...prev, { role: 'user', content: msg }])
    setInput('')
    setLoading(true)

    try {
      const res  = await fetch(`${BACKEND_URL}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: msg }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '**Error:** Could not reach the backend. Make sure it is running on port 8000.',
      }])
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const showSuggestions = messages.length === 1

  return (
    <div className="app">

      {/* ── Header ── */}
      <header>
        <div className="header-logo">⚡</div>
        <div className="header-text">
          <span className="header-title">Rate Limiter Agent</span>
          <span className="header-sub">LangGraph · RAG · Groq</span>
        </div>
      </header>

      {/* ── Messages ── */}
      <main>
        {messages.map((msg, i) => (
          <div key={i} className={`row ${msg.role}`}>
            <div className="avatar">{msg.role === 'assistant' ? '🤖' : '👤'}</div>
            <div className="bubble">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <div className="avatar">🤖</div>
            <div className="bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}

        {showSuggestions && (
          <div className="suggestions">
            {SUGGESTED.map(q => (
              <button key={q} className="suggestion-chip" onClick={() => send(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* ── Input ── */}
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
