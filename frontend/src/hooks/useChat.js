import { useState, useCallback } from 'react'
import api from '../api/axios'

function makeId() { return Math.random().toString(36).slice(2) }

function timestamp() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

export function useChat(conversationId) {
  const [messages, setMessages] = useState([
    {
      id: 'sys-0', role: 'assistant', agent: 'System',
      content: "Hello! I'm your medical assistant. Sign in and start a conversation to begin.",
      time: timestamp(), needsValidation: false,
    },
  ])
  const [loading, setLoading] = useState(false)
  const [pendingImage, setPendingImage] = useState(null) // { file, dataUrl }

  const loadConversation = useCallback(async (convId) => {
    if (!convId) return
    try {
      const res = await api.get(`/api/conversations/${convId}`)
      const msgs = res.data.messages.map(m => ({
        id: String(m.id),
        role: m.role,
        agent: m.agent_name || 'System',
        content: m.content,
        resultImage: m.result_image,
        time: new Date(m.created_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        needsValidation: false,
      }))
      setMessages(msgs.length ? msgs : [])
    } catch { /* ignore */ }
  }, [])

  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev, { id: makeId(), time: timestamp(), ...msg }])
  }, [])

  const sendMessage = useCallback(async (text) => {
    if (!conversationId) return
    addMessage({ role: 'user', content: text, agent: 'You', image: pendingImage?.dataUrl })
    const imageFile = pendingImage?.file
    setPendingImage(null)
    setLoading(true)

    try {
      let data
      if (imageFile) {
        const form = new FormData()
        form.append('image', imageFile)
        form.append('text', text)
        const res = await api.post(`/upload?conversation_id=${conversationId}`, form)
        data = res.data
      } else {
        const res = await api.post(`/chat?conversation_id=${conversationId}`, { query: text })
        data = res.data
      }

      addMessage({
        role: 'assistant',
        agent: data.agent || 'System',
        content: data.response,
        resultImage: data.result_image,
        needsValidation: (data.agent || '').includes('HUMAN_VALIDATION'),
      })
    } catch (err) {
      addMessage({ role: 'assistant', agent: 'System', content: `Error: ${err.response?.data?.detail || err.message}` })
    } finally {
      setLoading(false)
    }
  }, [conversationId, pendingImage, addMessage])

  const sendValidation = useCallback(async (validationResult, comments) => {
    if (!conversationId) return
    setLoading(true)
    try {
      const form = new FormData()
      form.append('validation_result', validationResult)
      if (comments) form.append('comments', comments)
      const res = await api.post(`/validate?conversation_id=${conversationId}`, form)
      const data = res.data
      addMessage({
        role: 'assistant',
        agent: 'HUMAN_VALIDATED',
        content: `${data.message}\n\n${data.response}`,
        needsValidation: false,
      })
    } catch (err) {
      addMessage({ role: 'assistant', agent: 'System', content: `Validation error: ${err.message}` })
    } finally {
      setLoading(false)
    }
  }, [conversationId, addMessage])

  const clearMessages = useCallback(() => setMessages([]), [])

  return {
    messages, loading, pendingImage, setPendingImage,
    sendMessage, sendValidation, loadConversation, clearMessages, addMessage,
  }
}
