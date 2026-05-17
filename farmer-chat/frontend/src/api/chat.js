const BASE = ''

/**
 * Streaming chat — yields { type, content/...metadata } objects.
 * Falls back to the legacy full-response endpoint if streaming fails.
 */
export async function* streamChat(email, query) {
  const res = await fetch(`${BASE}/api/chat/stream/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email_id: email, query }),
  })

  if (!res.ok || !res.body) {
    // Fallback to non-streaming endpoint
    const data = await res.json().catch(() => ({}))
    yield { type: 'metadata', ...data }
    yield { type: 'token', content: data.response || 'Could not get a response.' }
    yield { type: 'done' }
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete line

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const raw = line.slice(6).trim()
        if (!raw || raw === '[DONE]') continue
        try {
          yield JSON.parse(raw)
        } catch {
          // skip malformed chunks
        }
      }
    }
  }
}

export async function synthesiseAudio(email, text) {
  const res = await fetch(`${BASE}/api/chat/synthesise_audio/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email_id: email, text }),
  })
  return res.json()
}
