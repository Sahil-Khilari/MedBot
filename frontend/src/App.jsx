import { useState, useEffect, useCallback } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBars, faStethoscope } from '@fortawesome/free-solid-svg-icons'
import { useAuth } from './hooks/useAuth'
import { useChat } from './hooks/useChat'
import LoginPage from './pages/LoginPage'
import Sidebar from './components/Sidebar'
import ChatBody from './components/ChatBody'
import ChatInput from './components/ChatInput'
import api from './api/axios'

function downloadChat(messages) {
  const lines = ['Medical Assistant Conversation', `Exported: ${new Date().toLocaleString()}`, '='.repeat(40)]
  messages.forEach(m => {
    lines.push(`\n[${m.role.toUpperCase()}] ${m.time || ''}\n${m.content}`)
  })
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `medical-chat-${Date.now()}.txt`
  a.click()
  URL.revokeObjectURL(a.href)
}

export default function App() {
  const { user, loading: authLoading, logout } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)

  const { messages, loading, pendingImage, setPendingImage, sendMessage, sendValidation, loadConversation, clearMessages } = useChat(activeConvId)

  // Fetch conversation list when user logs in
  useEffect(() => {
    if (!user) { setConversations([]); setActiveConvId(null); return }
    api.get('/api/conversations')
      .then(res => {
        setConversations(res.data)
        if (res.data.length > 0 && !activeConvId) {
          setActiveConvId(res.data[0].id)
          loadConversation(res.data[0].id)
        }
      })
      .catch(() => {})
  }, [user])

  // Load messages when conversation switches
  useEffect(() => {
    if (activeConvId) loadConversation(activeConvId)
  }, [activeConvId])

  const createNewConversation = useCallback(async () => {
    try {
      const res = await api.post('/api/conversations', { title: 'New Conversation' })
      setConversations(prev => [res.data, ...prev])
      setActiveConvId(res.data.id)
      clearMessages()
    } catch { /* ignore */ }
  }, [clearMessages])

  const handleSelect = useCallback((id) => {
    setActiveConvId(id)
  }, [])

  const handleDelete = useCallback((id) => {
    setConversations(prev => prev.filter(c => c.id !== id))
    if (activeConvId === id) {
      setActiveConvId(null)
      clearMessages()
    }
  }, [activeConvId, clearMessages])

  const handleSend = useCallback((text) => {
    if (!activeConvId) {
      alert('Select or create a conversation first.')
      return
    }
    sendMessage(text)
    // Update conversation title in sidebar after first message
    setConversations(prev => prev.map(c =>
      c.id === activeConvId
        ? { ...c, title: c.title === 'New Conversation' ? text.slice(0, 60) : c.title, updated_at: new Date().toISOString() }
        : c
    ).sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)))
  }, [activeConvId, sendMessage])

  const handleImageDrop = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = ev => setPendingImage({ file, dataUrl: ev.target.result })
    reader.readAsDataURL(file)
  }, [setPendingImage])

  // Auth loading spinner
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  // Not logged in
  if (!user) return <LoginPage />

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        activeId={activeConvId}
        onSelect={handleSelect}
        onNew={createNewConversation}
        onDeleteConv={handleDelete}
        onClear={clearMessages}
        onDownload={() => downloadChat(messages)}
        user={user}
        onLogout={logout}
      />

      <main className="flex-1 flex flex-col min-w-0 h-screen">
        {/* Top bar */}
        <header className="flex-shrink-0 bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3">
          <button
            className="md:hidden text-slate-500 hover:text-slate-700"
            onClick={() => setSidebarOpen(true)}
          >
            <FontAwesomeIcon icon={faBars} />
          </button>
          <FontAwesomeIcon icon={faStethoscope} className="text-sky-500" />
          <div>
            <h2 className="text-sm font-semibold text-slate-800 leading-tight">Multi-Agent Medical Assistant</h2>
            {activeConvId && (
              <p className="text-xs text-slate-400">
                {conversations.find(c => c.id === activeConvId)?.title || 'Conversation'}
              </p>
            )}
          </div>
        </header>

        {/* No conversation selected */}
        {!activeConvId && (
          <div className="flex-1 flex items-center justify-center flex-col gap-3 text-slate-400">
            <p className="text-4xl">💬</p>
            <p className="text-sm">No conversation selected</p>
            <button
              onClick={createNewConversation}
              className="px-4 py-2 bg-sky-500 hover:bg-sky-600 text-white text-sm rounded-xl transition-colors"
            >
              Start New Conversation
            </button>
          </div>
        )}

        {/* Chat area */}
        {activeConvId && (
          <>
            <ChatBody
              messages={messages}
              loading={loading}
              onValidate={sendValidation}
              onDrop={handleImageDrop}
            />
            <ChatInput
              onSend={handleSend}
              pendingImage={pendingImage}
              onImageSelect={setPendingImage}
              onImageRemove={() => setPendingImage(null)}
              disabled={loading}
            />
          </>
        )}
      </main>
    </div>
  )
}
