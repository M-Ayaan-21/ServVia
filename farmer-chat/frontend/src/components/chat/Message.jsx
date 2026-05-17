import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faVolumeUp, faStop, faSpinner } from '@fortawesome/free-solid-svg-icons'
import AgentFlow from './AgentFlow'
import PipelineBadge from './PipelineBadge'
import TrustCard from './TrustCard'
import ChronoCard from './ChronoCard'
import SafetyCard from './SafetyCard'

export default function Message({ msg, msgRef, speakingId, onToggleSpeech }) {
  const isUser      = msg.role === 'user'
  const isStreaming  = msg.streaming === true
  const id           = msg.id
  const isLoadingAudio = speakingId === id + '_loading'
  const isSpeaking     = speakingId === id

  // ── User message ────────────────────────────────────────────────────────
  if (isUser) {
    return (
      <div ref={msgRef} className="flex justify-end mb-4">
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 380, damping: 30 }}
          className="px-5 py-3 text-white text-sm leading-relaxed"
          style={{
            background: 'linear-gradient(135deg,#dc2626,#991b1b)',
            borderRadius: '18px 18px 4px 18px',
            boxShadow: '0 2px 12px rgba(220,38,38,0.22)',
            maxWidth: '55%',
            wordBreak: 'break-word',
          }}
        >
          {msg.isJSX ? msg.content : msg.content}
        </motion.div>
      </div>
    )
  }

  // ── Bot message ─────────────────────────────────────────────────────────
  return (
    <div ref={msgRef} className="flex justify-start mb-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 340, damping: 30 }}
        style={{ width: '100%' }}
      >
        {/* Content bubble */}
        <div
          className="px-5 py-4 text-sm leading-relaxed"
          style={{
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid rgba(220,38,38,0.12)',
            borderRadius: '0 18px 18px 18px',
            borderLeft: isStreaming
              ? '2px solid rgba(220,38,38,0.5)'
              : '2px solid rgba(220,38,38,0.12)',
            transition: 'border-left-color 0.4s ease',
          }}
        >
          {/* ── Text content ── */}
          {isStreaming ? (
            /*
             * During streaming: plain text only.
             * ReactMarkdown re-parses every token which causes flicker
             * and layout shifts from broken markdown syntax mid-stream.
             */
            <div
              className="stream-cursor"
              style={{
                color: 'rgba(255,255,255,0.88)',
                lineHeight: 1.8,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                minHeight: '1.5em',
              }}
            >
              {msg.content}
            </div>
          ) : (
            <div className="prose-servvia">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content || ''}
              </ReactMarkdown>
            </div>
          )}

          {/* ── Metadata (fades in after streaming completes) ── */}
          {!isStreaming && msg.metadata && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
            >
              <AgentFlow blocked={msg.metadata.is_emergency_block} />

              {(msg.metadata.pipeline || msg.metadata.is_emergency_block) && (
                <div
                  className="flex flex-wrap gap-1.5 mt-3 pt-3"
                  style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
                >
                  {msg.metadata.is_emergency_block ? (
                    <PipelineBadge type="emergency_triage" />
                  ) : msg.metadata.pipeline === 'safe_response' ? (
                    <PipelineBadge type="safe" />
                  ) : msg.metadata.pipeline === 'emergency_intercept' ? (
                    <PipelineBadge type="emergency_intercept" />
                  ) : msg.metadata.pipeline === 'safety_blocked' ? (
                    <PipelineBadge type="safety_blocked" />
                  ) : null}
                </div>
              )}

              {msg.metadata.has_remedies !== false && (
                <TrustCard trust={msg.metadata.trust_verification} />
              )}
              <ChronoCard bioState={msg.metadata.bio_state} />
              <SafetyCard safety={msg.metadata.safety} />
            </motion.div>
          )}

          {/* ── Speaker button ── */}
          {!isStreaming && msg.content && (
            <div
              className="flex justify-end mt-3 pt-2.5"
              style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
            >
              <motion.button
                whileHover={{ scale: 1.08, background: 'rgba(220,38,38,0.14)' }}
                whileTap={{ scale: 0.93 }}
                onClick={() => onToggleSpeech(msg.content, id)}
                className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer text-xs"
                style={{
                  background: isSpeaking ? 'rgba(220,38,38,0.22)' : 'rgba(220,38,38,0.07)',
                  border: '1px solid rgba(220,38,38,0.18)',
                  color: isSpeaking ? '#dc2626' : 'rgba(255,255,255,0.4)',
                  animation: isSpeaking ? 'speakerPulse 1.5s infinite' : 'none',
                }}
                title={isSpeaking ? 'Stop' : isLoadingAudio ? 'Loading...' : 'Listen'}
              >
                <FontAwesomeIcon
                  icon={isLoadingAudio ? faSpinner : isSpeaking ? faStop : faVolumeUp}
                  spin={isLoadingAudio}
                />
              </motion.button>
            </div>
          )}
        </div>
      </motion.div>

      <style>{`
        @keyframes speakerPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.3); }
          50%       { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
        }
      `}</style>
    </div>
  )
}
