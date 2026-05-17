import { useRef, useState, useCallback } from 'react'

export function useVoiceInput(onResult) {
  const [isRecording, setIsRecording] = useState(false)
  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const recognitionRef = useRef(null)
  const finalRef = useRef('')

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      finalRef.current = ''
      setIsRecording(true)

      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition
        const rec = new SR()
        rec.continuous = true
        rec.interimResults = true
        rec.lang = 'en-US'
        rec.onresult = (e) => {
          let interim = '', confirmed = ''
          for (let i = e.resultIndex; i < e.results.length; i++) {
            if (e.results[i].isFinal) confirmed += e.results[i][0].transcript
            else interim += e.results[i][0].transcript
          }
          finalRef.current += confirmed
          onResult(finalRef.current + interim, false)
        }
        rec.start()
        recognitionRef.current = rec
      }

      const recorder = new MediaRecorder(stream)
      recorder.onstop = () => {
        if (recognitionRef.current) recognitionRef.current.stop()
        stream.getTracks().forEach(t => t.stop())
        streamRef.current = null
        setIsRecording(false)
        const text = finalRef.current.trim()
        if (text) onResult(text, true)
      }
      recorder.start()
      recorderRef.current = recorder
    } catch (err) {
      console.error('Mic error:', err)
      setIsRecording(false)
    }
  }, [onResult])

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
    }
  }, [])

  const toggle = useCallback(() => {
    isRecording ? stop() : start()
  }, [isRecording, start, stop])

  return { isRecording, toggle }
}
