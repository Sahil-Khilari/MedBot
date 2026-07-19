import { useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faChevronLeft, faChevronRight, faTimes, faDownload, faTrash } from '@fortawesome/free-solid-svg-icons'
import ConversationList from './ConversationList'
import UserProfile from './UserProfile'

const AGENTS = [
  { label: 'Conversation Agent', desc: 'General medical Q&A', color: '#dcfce7', text: '#166534' },
  { label: 'RAG Agent', desc: 'Document-based answers', color: '#dbeafe', text: '#1e40af' },
  { label: 'Web Search Agent', desc: 'Live medical info', color: '#ffedd5', text: '#9a3412' },
]
const CV_AGENTS = [
  { label: 'Brain Tumor', desc: 'MRI classification + segmentation', color: '#fee2e2', text: '#991b1b' },
  { label: 'Chest X-ray', desc: 'COVID-19 detection', color: '#f3e8ff', text: '#6b21a8' },
  { label: 'Skin Lesion', desc: 'Lesion segmentation', color: '#ccfbf1', text: '#115e59' },
]

export default function Sidebar({ isOpen, onClose, conversations, activeId, onSelect, onNew, onDeleteConv, onClear, onDownload, user, onLogout }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/40 z-20 md:hidden" onClick={onClose} />
      )}

      <aside
        className={`
          flex flex-col flex-shrink-0 h-full bg-sidebar border-r border-sidebar-border
          transition-all duration-300 z-30
          ${isOpen ? 'fixed left-0 top-0 bottom-0 w-72' : 'hidden md:flex'}
          ${collapsed ? 'md:w-16' : 'md:w-72'}
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-sidebar-border flex-shrink-0">
          {!collapsed && (
            <div>
              <h1 className="text-white font-bold text-base leading-tight">MedAssist</h1>
              <p className="text-slate-500 text-xs">Multi-Agent System</p>
            </div>
          )}
          <button onClick={() => setCollapsed(c => !c)} className="text-slate-500 hover:text-slate-300 transition-colors md:block hidden">
            <FontAwesomeIcon icon={collapsed ? faChevronRight : faChevronLeft} className="text-xs" />
          </button>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 md:hidden">
            <FontAwesomeIcon icon={faTimes} />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto sidebar-scroll py-3 px-2 space-y-4">
          {!collapsed && (
            <>
              {/* Conversations */}
              <ConversationList
                conversations={conversations}
                activeId={activeId}
                onSelect={id => { onSelect(id); onClose() }}
                onNew={onNew}
                onDelete={onDeleteConv}
              />

              {/* Agent info */}
              <div>
                <p className="text-xs font-semibold text-sidebar-border uppercase tracking-wider px-3 mb-1">Agents</p>
                {AGENTS.map(a => (
                  <div key={a.label} className="flex items-center gap-2 px-3 py-1.5">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: a.color }} />
                    <div>
                      <p className="text-xs text-slate-300">{a.label}</p>
                      <p className="text-[10px] text-slate-500">{a.desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <p className="text-xs font-semibold text-sidebar-border uppercase tracking-wider px-3 mb-1">Vision</p>
                {CV_AGENTS.map(a => (
                  <div key={a.label} className="flex items-center gap-2 px-3 py-1.5">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: a.color }} />
                    <div>
                      <p className="text-xs text-slate-300">{a.label}</p>
                      <p className="text-[10px] text-slate-500">{a.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Bottom controls */}
        {!collapsed && (
          <div className="flex-shrink-0 px-2 py-2 border-t border-sidebar-border space-y-1">
            <button
              onClick={onDownload}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-slate-200 hover:bg-sidebar-surface rounded-lg transition-colors"
            >
              <FontAwesomeIcon icon={faDownload} className="text-xs" />
              Download Chat
            </button>
            <button
              onClick={onClear}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-red-400 hover:bg-sidebar-surface rounded-lg transition-colors"
            >
              <FontAwesomeIcon icon={faTrash} className="text-xs" />
              Clear Conversation
            </button>
            <UserProfile user={user} onLogout={onLogout} />
          </div>
        )}
      </aside>
    </>
  )
}
