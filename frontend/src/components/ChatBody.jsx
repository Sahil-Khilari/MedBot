import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import ThinkingIndicator from './ThinkingIndicator'

export default function ChatBody({ messages, loading, onValidate, onDrop }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleDragOver = (e) => { e.preventDefault() }
  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && /\.(png|jpe?g)$/i.test(file.name)) onDrop?.(file)
  }

  return (
    <div
      className="flex-1 overflow-y-auto custom-scroll px-4 py-6 space-y-4 bg-slate-50"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {messages.length === 0 && !loading && (
        <div className="text-center text-slate-400 text-sm mt-20">
          <p className="text-4xl mb-3">🏥</p>
          <p>Start a conversation or upload a medical image</p>
          <p className="text-xs mt-1 text-slate-300">You can also drag & drop images here</p>
        </div>
      )}
      {messages.map(msg => (
        <MessageBubble key={msg.id} message={msg} onValidate={onValidate} />
      ))}
      {loading && <ThinkingIndicator />}
      <div ref={bottomRef} />
    </div>
  )
}
