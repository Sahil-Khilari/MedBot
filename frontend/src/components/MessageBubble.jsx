import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faVolumeUp, faSpinner, faPause, faRotateLeft } from '@fortawesome/free-solid-svg-icons'
import AgentBadge, { getAgentStyle } from './AgentBadge'
import ValidationCard from './ValidationCard'
import { useTTS } from '../hooks/useTTS'

function AgentAvatar({ agent }) {
  const style = getAgentStyle(agent)
  const initial = (style.label || 'S')[0].toUpperCase()
  return (
    <div
      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
      style={{ background: style.bg, color: style.text }}
    >
      {initial}
    </div>
  )
}

function TTSButton({ text }) {
  const { status, play, toggle } = useTTS()
  const icons = { idle: faVolumeUp, loading: faSpinner, playing: faPause, paused: faRotateLeft }
  const icon = icons[status] || faVolumeUp

  return (
    <button
      onClick={() => status === 'idle' ? play(text) : toggle()}
      className="mt-1 text-slate-400 hover:text-sky-500 transition-colors p-1"
      title="Play voice response"
    >
      <FontAwesomeIcon icon={icon} spin={status === 'loading'} className="text-sm" />
    </button>
  )
}

function ResultImages({ src, originalSrc, agent }) {
  const isSkinLesion = (agent || '').includes('SKIN_LESION')
  const [lightbox, setLightbox] = useState(null)

  if (isSkinLesion && originalSrc) {
    return (
      <div className="mt-3 flex gap-2">
        {[{ url: originalSrc, label: 'Original Image' }, { url: src, label: 'Segmentation Result' }].map(({ url, label }) => (
          <div key={label} className="flex-1 rounded-xl overflow-hidden border border-slate-200 shadow-sm cursor-pointer" onClick={() => setLightbox(url)}>
            <img src={url} alt={label} className="w-full h-32 object-contain bg-slate-50" />
            <div className="text-xs text-center py-1 text-slate-500 bg-white border-t border-slate-100">{label}</div>
          </div>
        ))}
        {lightbox && (
          <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center" onClick={() => setLightbox(null)}>
            <img src={lightbox} alt="Full view" className="max-h-[90vh] max-w-[90vw] rounded-xl shadow-2xl" />
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="mt-3">
      <img
        src={src}
        alt="Analysis result"
        className="max-h-64 rounded-xl border border-slate-200 shadow-sm cursor-pointer object-contain bg-slate-50"
        onClick={() => setLightbox(src)}
      />
      {lightbox && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Full view" className="max-h-[90vh] max-w-[90vw] rounded-xl shadow-2xl" />
        </div>
      )}
    </div>
  )
}

export default function MessageBubble({ message, onValidate }) {
  const { role, content, agent, time, resultImage, image, needsValidation } = message
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  if (role === 'user') {
    return (
      <div className="flex justify-end animate-slideIn">
        <div className="max-w-[75%]">
          <div className="flex items-center justify-end gap-2 mb-1">
            <span className="text-xs text-slate-400">{time}</span>
            <span className="text-xs text-slate-500 font-medium">You</span>
          </div>
          <div className="bg-sky-100 text-slate-800 rounded-2xl rounded-tr-none px-4 py-3 shadow-sm">
            {image && (
              <img src={image} alt="Uploaded" className="max-h-32 rounded-lg mb-2 object-contain" />
            )}
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 animate-slideIn">
      <AgentAvatar agent={agent} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <AgentBadge agent={agent} />
          <span className="text-xs text-slate-400">{time}</span>
        </div>
        <div className="bg-white rounded-2xl rounded-tl-none px-4 py-3 shadow-sm max-w-[85%]">
          <div className="msg-content text-sm text-slate-800 leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        </div>
        <TTSButton text={content} />
        {resultImage && (
          <ResultImages
            src={`${apiUrl}${resultImage}`}
            agent={agent}
          />
        )}
        {needsValidation && <ValidationCard onValidate={onValidate} />}
      </div>
    </div>
  )
}
