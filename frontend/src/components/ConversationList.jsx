import { useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faTrash, faComment } from '@fortawesome/free-solid-svg-icons'
import api from '../api/axios'

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export default function ConversationList({ conversations, activeId, onSelect, onNew, onDelete }) {
  const [hoverId, setHoverId] = useState(null)

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    try { await api.delete(`/api/conversations/${id}`) } catch { /* ignore */ }
    onDelete(id)
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between px-3 mb-1">
        <span className="text-xs font-semibold text-sidebar-border uppercase tracking-wider">Conversations</span>
        <button
          onClick={onNew}
          className="w-6 h-6 flex items-center justify-center text-slate-400 hover:text-sky-400 transition-colors"
          title="New conversation"
        >
          <FontAwesomeIcon icon={faPlus} className="text-xs" />
        </button>
      </div>

      {conversations.length === 0 && (
        <p className="text-xs text-slate-500 px-3 py-2">No conversations yet. Click + to start one.</p>
      )}

      {conversations.map(conv => (
        <div
          key={conv.id}
          onClick={() => onSelect(conv.id)}
          onMouseEnter={() => setHoverId(conv.id)}
          onMouseLeave={() => setHoverId(null)}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors group ${
            activeId === conv.id
              ? 'bg-sidebar-surface text-white'
              : 'text-slate-400 hover:bg-sidebar-surface hover:text-slate-200'
          }`}
        >
          <FontAwesomeIcon icon={faComment} className="text-xs flex-shrink-0 opacity-60" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium truncate">{conv.title}</p>
            <p className="text-[10px] opacity-50">{timeAgo(conv.updated_at)}</p>
          </div>
          {hoverId === conv.id && (
            <button
              onClick={(e) => handleDelete(e, conv.id)}
              className="text-slate-500 hover:text-red-400 transition-colors"
              title="Delete conversation"
            >
              <FontAwesomeIcon icon={faTrash} className="text-xs" />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
