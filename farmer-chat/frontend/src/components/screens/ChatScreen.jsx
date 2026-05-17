import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faHandHoldingMedical, faShieldAlt, faRobot, faSearch, faCheckCircle } from '@fortawesome/free-solid-svg-icons'
import Message from '../chat/Message'
import TypingIndicator from '../ui/TypingIndicator'

function parseArray(val) {
  if (Array.isArray(val)) return val
  if (typeof val === 'string') return val.split(',').map(s => s.trim()).filter(Boolean)
  return []
}

function WelcomeBanner({ profile }) {
  const allergies = parseArray(profile?.allergies)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 260, damping: 24 }}
      className="text-left py-10 px-2"
    >
      {/* Greeting row */}
      <div className="flex items-center gap-4 mb-4">
        <motion.div
          animate={{ y: [0, -6, 0] }}
          transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
          className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0 text-2xl"
          style={{ background: 'rgba(220,38,38,0.08)', border: '1px solid rgba(220,38,38,0.15)' }}
        >
          <FontAwesomeIcon icon={faHandHoldingMedical} style={{ color: '#dc2626' }} />
        </motion.div>
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight" style={{ lineHeight: 1.2 }}>
            Good to see you, {profile?.first_name}
          </h2>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>
            What can ServVia help you with today?
          </p>
        </div>
      </div>

      {allergies.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl mb-5 text-xs"
          style={{
            background: 'rgba(220,38,38,0.07)',
            border: '1px solid rgba(220,38,38,0.18)',
            color: '#dc2626',
          }}
        >
          <FontAwesomeIcon icon={faShieldAlt} className="text-xs" />
          Allergen protection active · {allergies.join(', ')}
        </motion.div>
      )}

      <motion.div
        className="flex items-center gap-2 flex-wrap"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25 }}
      >
        {[
          { icon: faRobot,       label: 'Proposer', color: '#60a5fa', bg: 'rgba(59,130,246,0.08)',  border: 'rgba(59,130,246,0.2)'  },
          null,
          { icon: faSearch,      label: 'Critic',   color: '#c084fc', bg: 'rgba(168,85,247,0.08)', border: 'rgba(168,85,247,0.2)' },
          null,
          { icon: faCheckCircle, label: 'Verified', color: '#22c55e', bg: 'rgba(34,197,94,0.08)',  border: 'rgba(34,197,94,0.2)'  },
        ].map((item, i) =>
          item === null ? (
            <span key={i} className="text-xs" style={{ color: 'rgba(255,255,255,0.15)' }}>→</span>
          ) : (
            <span key={i}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium"
              style={{ background: item.bg, border: `1px solid ${item.border}`, color: item.color }}
            >
              <FontAwesomeIcon icon={item.icon} className="text-xs" />
              {item.label}
            </span>
          )
        )}
      </motion.div>
    </motion.div>
  )
}

export default function ChatScreen({ messages, isThinking, profile, speakingId, onToggleSpeech }) {
  const containerRef = useRef(null)
  const msgRefs     = useRef({})
  const prevStreamingIdRef = useRef(null)

  // Which message is currently streaming?
  const streamingMsgId = [...messages].reverse().find(m => m.streaming)?.id ?? null

  // ── Scroll rule 1: follow bottom on every message update (covers streaming tokens)
  useEffect(() => {
    if (!streamingMsgId) return
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, streamingMsgId])

  // ── Scroll rule 2: scroll to bottom when thinking indicator appears
  useEffect(() => {
    if (!isThinking) return
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [isThinking])

  // ── Scroll rule 3: when streaming ends, scroll to TOP of that response
  useEffect(() => {
    const prev = prevStreamingIdRef.current
    prevStreamingIdRef.current = streamingMsgId

    if (prev && !streamingMsgId) {
      // Give ReactMarkdown ~200 ms to render before measuring position
      setTimeout(() => {
        const el = msgRefs.current[prev]
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 200)
    }
  }, [streamingMsgId])

  return (
    <div
      ref={containerRef}
      className="h-full overflow-y-auto scrollbar-red"
    >
      <div className="px-6 pb-4" style={{ maxWidth: '920px', margin: '0 auto' }}>
        <WelcomeBanner profile={profile} />

        <AnimatePresence initial={false}>
          {messages.map(msg => (
            <Message
              key={msg.id}
              msg={msg}
              msgRef={el => { if (el) msgRefs.current[msg.id] = el }}
              speakingId={speakingId}
              onToggleSpeech={onToggleSpeech}
            />
          ))}
        </AnimatePresence>

        <AnimatePresence>
          {isThinking && (
            <motion.div
              key="thinking"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.2 }}
              className="flex justify-start mb-4"
            >
              <div style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(220,38,38,0.12)',
                borderRadius: '0 18px 18px 18px',
              }}>
                <TypingIndicator />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
