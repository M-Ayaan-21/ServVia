import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ParticleCanvas from './components/ParticleCanvas'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import EmailScreen from './components/screens/EmailScreen'
import OnboardingScreen from './components/screens/OnboardingScreen'
import ChatScreen from './components/screens/ChatScreen'
import InputBar from './components/input/InputBar'
import { checkProfile, saveProfile } from './api/profile'
import { streamChat } from './api/chat'
import { analyzeSkin } from './api/skin'
import { analyzeLabReport } from './api/labReport'
import { useSpeech } from './hooks/useSpeech'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faFileMedical, faImage, faFilePdf } from '@fortawesome/free-solid-svg-icons'

let msgCounter = 0
function newId() { return `msg_${++msgCounter}` }

const screenVariants = {
  initial: { opacity: 0, x: 30 },
  animate: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 300, damping: 28 } },
  exit:    { opacity: 0, x: -30, transition: { duration: 0.18 } },
}

export default function App() {
  const [screen, setScreen] = useState('email')
  const [userEmail, setUserEmail] = useState('')
  const [userProfile, setUserProfile] = useState(null)
  const [messages, setMessages] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const [loading, setLoading] = useState(false)
  const { speakingId, toggleSpeech } = useSpeech(userEmail)

  // ─── Profile flow ────────────────────────────────────────────────────────

  async function handleEmailContinue(email) {
    setLoading(true)
    try {
      const data = await checkProfile(email)
      setUserEmail(email)
      if (data.exists && data.is_complete) {
        setUserProfile(data.profile)
        setScreen('chat')
      } else {
        setScreen('onboarding')
      }
    } catch {
      alert('Connection error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function handleSaveProfile({ name, allergies, conditions, medications }) {
    setLoading(true)
    try {
      const data = await saveProfile({ email: userEmail, name, allergies, conditions, medications })
      if (data.success) {
        setUserProfile(data.profile)
        setScreen('chat')
      } else {
        alert('Error saving profile. Please try again.')
      }
    } catch {
      alert('Connection error.')
    } finally {
      setLoading(false)
    }
  }

  // ─── Chat / streaming ────────────────────────────────────────────────────

  function addMessage(msg) {
    setMessages(prev => [...prev, msg])
  }

  function patchMessage(id, updater) {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, ...updater(m) } : m))
  }

  const handleSend = useCallback(async (text) => {
    if (!text.trim() || isThinking) return

    addMessage({ id: newId(), role: 'user', content: text })
    setIsThinking(true)

    const botId = newId()
    let metadata = null
    let started = false

    try {
      for await (const chunk of streamChat(userEmail, text)) {
        if (chunk.type === 'metadata') {
          metadata = chunk
          if (!started) {
            setIsThinking(false)
            addMessage({ id: botId, role: 'bot', content: '', streaming: true, metadata: null })
            started = true
          }
        } else if (chunk.type === 'token') {
          if (!started) {
            setIsThinking(false)
            addMessage({ id: botId, role: 'bot', content: '', streaming: true, metadata: null })
            started = true
          }
          patchMessage(botId, m => ({ content: m.content + chunk.content }))
        } else if (chunk.type === 'done') {
          patchMessage(botId, () => ({ streaming: false, metadata }))
        }
      }
    } catch (err) {
      console.error(err)
      setIsThinking(false)
      if (!started) {
        addMessage({ id: botId, role: 'bot', content: 'Connection error. Please try again.', streaming: false, metadata: null })
      } else {
        patchMessage(botId, m => ({ content: m.content || 'Connection error.', streaming: false, metadata }))
      }
    } finally {
      setIsThinking(false)
    }
  }, [userEmail, isThinking])

  // ─── Skin analysis ───────────────────────────────────────────────────────

  const handleSkinImage = useCallback(async (file) => {
    const preview = URL.createObjectURL(file)
    addMessage({
      id: newId(), role: 'user', isJSX: true,
      content: (
        <div className="text-center">
          <img src={preview} alt="Skin" style={{ maxWidth: '300px', maxHeight: '300px', borderRadius: '14px', marginBottom: '0.75rem' }} />
          <p>Skin image uploaded</p>
        </div>
      ),
    })
    setIsThinking(true)
    try {
      const data = await analyzeSkin(userEmail, file)
      setIsThinking(false)
      addMessage({
        id: newId(), role: 'bot', streaming: false, metadata: null,
        content: data.success ? data.recommendations : `Error: ${data.error}\n\n${data.suggestion || ''}`,
      })
    } catch {
      setIsThinking(false)
      addMessage({ id: newId(), role: 'bot', content: 'Connection error.', streaming: false, metadata: null })
    }
  }, [userEmail])

  // ─── Lab report ──────────────────────────────────────────────────────────

  const handleLabFiles = useCallback(async (files) => {
    addMessage({
      id: newId(), role: 'user', isJSX: true,
      content: (
        <div className="text-center">
          <FontAwesomeIcon icon={faFileMedical} style={{ color: '#dc2626', fontSize: '3rem', marginBottom: '0.5rem' }} />
          <p style={{ marginBottom: '0.5rem' }}>Uploaded {files.length} file{files.length > 1 ? 's' : ''}:</p>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {files.map((f, i) => (
              <li key={i} style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>
                <FontAwesomeIcon
                  icon={f.type.includes('pdf') ? faFilePdf : faImage}
                  style={{ color: '#dc2626', marginRight: '0.4rem' }}
                />
                {f.name}
              </li>
            ))}
          </ul>
        </div>
      ),
    })
    setIsThinking(true)
    try {
      const data = await analyzeLabReport(userEmail, files)
      setIsThinking(false)
      let content = data.success ? (data.summary || data.formatted_summary || '') : `Error: ${data.error}`
      if (content && !content.includes('Medical Disclaimer')) {
        content += '\n\n---\n\n⚠️ **Important Medical Disclaimer:** This AI-generated analysis is for educational purposes only.'
      }
      addMessage({ id: newId(), role: 'bot', content, streaming: false, metadata: null })
    } catch {
      setIsThinking(false)
      addMessage({ id: newId(), role: 'bot', content: 'Connection error.', streaming: false, metadata: null })
    }
  }, [userEmail])

  // ─── Render ───────────────────────────────────────────────────────────────

  const isChat = screen === 'chat'

  return (
    <div className="relative w-full h-full flex flex-col" style={{ background: '#050505' }}>
      <ParticleCanvas />

      {/* ── Auth screens (email / onboarding) ── full-screen centered */}
      {!isChat && (
        <div className="relative flex flex-col h-full" style={{ zIndex: 1 }}>
          <Header profile={userProfile} onEditProfile={() => setScreen('onboarding')} minimal />

          <div className="flex-1 overflow-hidden relative">
            <AnimatePresence mode="wait">
              {screen === 'email' && (
                <motion.div key="email" variants={screenVariants} initial="initial" animate="animate" exit="exit"
                  className="absolute inset-0 overflow-y-auto scrollbar-red">
                  <div className="mx-auto px-6" style={{ maxWidth: '600px' }}>
                    <EmailScreen onContinue={handleEmailContinue} loading={loading} />
                  </div>
                </motion.div>
              )}
              {screen === 'onboarding' && (
                <motion.div key="onboarding" variants={screenVariants} initial="initial" animate="animate" exit="exit"
                  className="absolute inset-0 overflow-y-auto scrollbar-red">
                  <div className="mx-auto px-6" style={{ maxWidth: '560px' }}>
                    <OnboardingScreen
                      isEdit={!!userProfile}
                      existingProfile={userProfile}
                      onSave={handleSaveProfile}
                      loading={loading}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* ── Chat screen: sidebar + full-width main ── */}
      {isChat && (
        <div className="relative flex h-full" style={{ zIndex: 1 }}>
          {/* Sidebar */}
          <Sidebar
            profile={userProfile}
            onQuickQuestion={handleSend}
            onEditProfile={() => setScreen('onboarding')}
          />

          {/* Main content area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header
              profile={userProfile}
              onEditProfile={() => setScreen('onboarding')}
              compact
            />

            <div className="flex-1 overflow-hidden relative">
              <motion.div
                key="chat"
                variants={screenVariants}
                initial="initial"
                animate="animate"
                className="absolute inset-0"
              >
                <ChatScreen
                  messages={messages}
                  isThinking={isThinking}
                  profile={userProfile}
                  speakingId={speakingId}
                  onToggleSpeech={toggleSpeech}
                />
              </motion.div>
            </div>

            <InputBar
              onSend={handleSend}
              onSkinImage={handleSkinImage}
              onLabFiles={handleLabFiles}
              disabled={isThinking}
            />
          </div>
        </div>
      )}
    </div>
  )
}
