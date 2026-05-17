import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faMicrophone, faStop, faPaperPlane } from '@fortawesome/free-solid-svg-icons'
import { useVoiceInput } from '../../hooks/useVoiceInput'
import OptionsMenu from './OptionsMenu'

export default function InputBar({ onSend, onSkinImage, onLabFiles, disabled }) {
  const [text, setText]       = useState('')
  const [menuOpen, setMenuOpen] = useState('')
  const [focused, setFocused]  = useState(false)
  const inputRef = useRef()

  const handleVoiceResult = useCallback((transcript, isFinal) => {
    setText(transcript)
    if (isFinal && transcript.trim()) {
      setTimeout(() => { onSend(transcript.trim()); setText('') }, 100)
    }
  }, [onSend])

  const { isRecording, toggle: toggleMic } = useVoiceInput(handleVoiceResult)

  function submit() {
    const msg = text.trim()
    if (!msg || disabled) return
    onSend(msg)
    setText('')
    inputRef.current?.focus()
  }

  const hasText = text.trim().length > 0

  const iconBtnBase = {
    width: '36px', height: '36px',
    borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: 'none', cursor: 'pointer', flexShrink: 0,
    fontSize: '0.9rem',
    transition: 'all 0.2s',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 28 }}
      className="relative px-4 pb-5 pt-2"
      style={{ maxWidth: '920px', margin: '0 auto', width: '100%' }}
    >
      {/* Options menu */}
      <OptionsMenu
        open={menuOpen === 'open'}
        onClose={() => setMenuOpen('')}
        onSkinImage={f  => { onSkinImage(f);  setMenuOpen('') }}
        onLabFiles={fs  => { onLabFiles(fs);  setMenuOpen('') }}
      />
      {menuOpen === 'open' && (
        <div className="fixed inset-0 z-10" onClick={() => setMenuOpen('')} />
      )}

      {/* Input shell */}
      <div
        className="flex items-center gap-2 px-3"
        style={{
          background: focused
            ? 'rgba(255,255,255,0.04)'
            : 'rgba(255,255,255,0.025)',
          border: focused
            ? '1px solid rgba(220,38,38,0.35)'
            : '1px solid rgba(255,255,255,0.07)',
          borderRadius: '24px',
          boxShadow: focused ? '0 0 0 3px rgba(220,38,38,0.07)' : 'none',
          transition: 'all 0.25s ease',
          position: 'relative',
          zIndex: 15,
        }}
      >
        {/* Plus */}
        <motion.button
          whileHover={{ background: 'rgba(255,255,255,0.08)', color: '#dc2626' }}
          whileTap={{ scale: 0.88 }}
          onClick={e => { e.stopPropagation(); setMenuOpen(o => o === 'open' ? '' : 'open') }}
          style={{ ...iconBtnBase, background: 'transparent', color: 'rgba(255,255,255,0.35)' }}
          title="Attachments"
        >
          <FontAwesomeIcon icon={faPlus} />
        </motion.button>

        {/* Text */}
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && submit()}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={isRecording ? 'Listening…' : 'Ask about symptoms, remedies, medications…'}
          disabled={disabled}
          autoComplete="off"
          className="flex-1 bg-transparent border-none outline-none text-white py-3.5 px-1 text-sm"
          style={{
            fontFamily: 'inherit',
            caretColor: '#dc2626',
            letterSpacing: '0.01em',
          }}
        />

        {/* Mic */}
        <motion.button
          whileHover={{ background: isRecording ? undefined : 'rgba(255,255,255,0.08)' }}
          whileTap={{ scale: 0.88 }}
          onClick={toggleMic}
          style={{
            ...iconBtnBase,
            background: isRecording ? 'rgba(220,38,38,0.9)' : 'transparent',
            color: isRecording ? '#fff' : 'rgba(255,255,255,0.35)',
            animation: isRecording ? 'pulseRec 1.5s infinite' : 'none',
          }}
          title={isRecording ? 'Stop' : 'Voice input'}
        >
          <FontAwesomeIcon icon={isRecording ? faStop : faMicrophone} />
        </motion.button>

        {/* Send */}
        <AnimatePresence>
          {hasText && (
            <motion.button
              key="send"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 500, damping: 26 }}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.92 }}
              onClick={submit}
              disabled={disabled}
              style={{
                ...iconBtnBase,
                background: 'linear-gradient(135deg,#dc2626,#991b1b)',
                color: '#fff',
                boxShadow: '0 2px 10px rgba(220,38,38,0.3)',
                opacity: disabled ? 0.5 : 1,
              }}
            >
              <FontAwesomeIcon icon={faPaperPlane} style={{ fontSize: '0.8rem' }} />
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      {/* Hint */}
      <p className="text-center mt-2 text-xs" style={{ color: 'rgba(255,255,255,0.18)' }}>
        ServVia checks every response for safety · Not a substitute for medical advice
      </p>

      <style>{`
        @keyframes pulseRec {
          0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
          50%      { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
        }
      `}</style>
    </motion.div>
  )
}
