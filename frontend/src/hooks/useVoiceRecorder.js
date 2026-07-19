import { useState, useRef, useCallback, useEffect } from 'react'
import api from '../api/axios'

export function useVoiceRecorder(onTranscript) {
  const [isRecording, setIsRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [status, setStatus] = useState('idle') // idle | recording | processing
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)

  const startTimer = () => {
    setSeconds(0)
    timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)
  }
  const stopTimer = () => {
    clearInterval(timerRef.current)
    timerRef.current = null
  }

  useEffect(() => () => stopTimer(), [])

  const timerDisplay = String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0')

  const toggle = useCallback(async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop()
      stopTimer()
      setIsRecording(false)
      setStatus('processing')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(t => t.stop())
        try {
          const form = new FormData()
          form.append('audio', blob, 'recording.webm')
          const res = await api.post('/transcribe', form)
          if (res.data?.transcript) onTranscript(res.data.transcript)
        } catch { /* ignore */ }
        setStatus('idle')
      }
      mr.start()
      mediaRecorderRef.current = mr
      setIsRecording(true)
      setStatus('recording')
      startTimer()
    } catch (e) {
      console.error('Microphone access denied', e)
    }
  }, [isRecording, onTranscript])

  return { isRecording, status, timerDisplay, toggle }
}
