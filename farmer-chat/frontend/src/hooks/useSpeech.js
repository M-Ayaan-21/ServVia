import { useRef, useState, useCallback } from 'react'
import { synthesiseAudio } from '../api/chat'

function cleanTextForSpeech(text) {
  return text
    .replace(/#{1,6}\s/g, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\|/g, ', ')
    .replace(/---+/g, '')
    .replace(/-\s/g, ', ')
    .replace(/\n+/g, '. ')
    .replace(/[🔬📋🏥📊🔍💊⚠️✅❌🟢🟡🔴👤🎯📚🧠💪🥗🩺📌📈❓🚨⚡🌿⏰💧📅🔗👋⚕️]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .substring(0, 3000)
}

export function useSpeech(userEmail) {
  const [speakingId, setSpeakingId] = useState(null)
  const audioRef = useRef(null)

  const stopSpeech = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (window.speechSynthesis) window.speechSynthesis.cancel()
    setSpeakingId(null)
  }, [])

  const browserTTS = useCallback((text, id) => {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text.substring(0, 1000))
    u.lang = 'en-IN'
    u.rate = 0.9
    u.onstart = () => setSpeakingId(id)
    u.onend = () => setSpeakingId(null)
    u.onerror = () => setSpeakingId(null)
    window.speechSynthesis.speak(u)
  }, [])

  const toggleSpeech = useCallback(async (text, id) => {
    if (speakingId === id) { stopSpeech(); return }
    if (speakingId) stopSpeech()

    const clean = cleanTextForSpeech(text)
    if (!clean) return

    setSpeakingId(id + '_loading')
    try {
      const data = await synthesiseAudio(userEmail, clean)
      if (data.audio && !data.error) {
        const bytes = atob(data.audio)
        const buf = new ArrayBuffer(bytes.length)
        const view = new Uint8Array(buf)
        for (let i = 0; i < bytes.length; i++) view[i] = bytes.charCodeAt(i)
        const blob = new Blob([buf], { type: 'audio/ogg' })
        const audio = new Audio(URL.createObjectURL(blob))
        audioRef.current = audio
        setSpeakingId(id)
        audio.onended = () => setSpeakingId(null)
        audio.onerror = () => { setSpeakingId(null); browserTTS(clean, id) }
        await audio.play()
      } else {
        browserTTS(clean, id)
      }
    } catch {
      browserTTS(clean, id)
    }
  }, [speakingId, userEmail, stopSpeech, browserTTS])

  return { speakingId, toggleSpeech, stopSpeech }
}
