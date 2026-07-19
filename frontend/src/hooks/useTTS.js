import { useState, useRef, useCallback } from 'react'
import api from '../api/axios'

function removeMarkdown(text) {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`]*`/g, '')
    .replace(/^#+\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/!\[.*?\]\(.*?\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^[-*+]\s+/gm, '')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/^>\s+/gm, '')
    .replace(/\n{2,}/g, '. ')
    .trim()
    .slice(0, 1000)
}

export function useTTS() {
  const [status, setStatus] = useState('idle') // idle | loading | playing | paused
  const audioRef = useRef(null)
  const urlRef = useRef(null)

  const play = useCallback(async (text) => {
    if (audioRef.current) {
      audioRef.current.pause()
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      audioRef.current = null
    }

    setStatus('loading')
    try {
      const res = await api.post('/generate-speech', { text: removeMarkdown(text), voice_id: 'default' }, { responseType: 'blob' })
      const url = URL.createObjectURL(res.data)
      urlRef.current = url
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onplay = () => setStatus('playing')
      audio.onpause = () => setStatus('paused')
      audio.onended = () => { setStatus('idle'); URL.revokeObjectURL(url); urlRef.current = null }
      audio.play()
    } catch {
      setStatus('idle')
    }
  }, [])

  const toggle = useCallback(() => {
    if (!audioRef.current) return
    if (status === 'playing') audioRef.current.pause()
    else audioRef.current.play()
  }, [status])

  return { status, play, toggle }
}
