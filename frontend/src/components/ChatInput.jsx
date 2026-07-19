import { useRef, useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPaperPlane, faPaperclip, faMicrophone, faStop, faSpinner } from '@fortawesome/free-solid-svg-icons'
import { useVoiceRecorder } from '../hooks/useVoiceRecorder'
import ImagePreviewStrip from './ImagePreviewStrip'

export default function ChatInput({ onSend, pendingImage, onImageSelect, onImageRemove, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  const { isRecording, status: voiceStatus, timerDisplay, toggle: toggleRecording } = useVoiceRecorder((transcript) => {
    setText(prev => prev ? prev + ' ' + transcript : transcript)
  })

  const resize = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed && !pendingImage) return
    onSend(trimmed)
    setText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const handleFile = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => onImageSelect({ file, dataUrl: ev.target.result })
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  const voiceBusy = voiceStatus === 'processing'

  return (
    <div className="flex-shrink-0 bg-white border-t border-slate-200">
      <ImagePreviewStrip image={pendingImage} onRemove={onImageRemove} />
      <div className="px-4 py-3 flex items-end gap-2">
        {/* File attach */}
        <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg" className="hidden" onChange={handleFile} />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="w-9 h-9 flex items-center justify-center text-slate-400 hover:text-sky-500 transition-colors"
          title="Attach image"
        >
          <FontAwesomeIcon icon={faPaperclip} />
        </button>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => { setText(e.target.value); resize() }}
          onKeyDown={handleKey}
          placeholder="Ask a medical question..."
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none border border-slate-200 rounded-2xl px-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-400 max-h-40 overflow-y-auto"
          style={{ minHeight: '42px' }}
        />

        {/* Voice */}
        <button
          onClick={toggleRecording}
          disabled={voiceBusy || disabled}
          className={`w-9 h-9 flex items-center justify-center rounded-full transition-all ${
            isRecording ? 'bg-red-500 text-white animate-pulse' : 'text-slate-400 hover:text-sky-500'
          }`}
          title={isRecording ? `Stop recording (${timerDisplay})` : 'Start voice input'}
        >
          <FontAwesomeIcon icon={voiceBusy ? faSpinner : isRecording ? faStop : faMicrophone} spin={voiceBusy} />
        </button>
        {isRecording && <span className="text-xs text-red-500 font-mono absolute right-20 bottom-5">{timerDisplay}</span>}

        {/* Send */}
        <button
          onClick={handleSend}
          disabled={disabled || (!text.trim() && !pendingImage)}
          className="w-9 h-9 flex items-center justify-center bg-sky-500 hover:bg-sky-600 disabled:opacity-40 text-white rounded-full transition-colors"
          title="Send message"
        >
          <FontAwesomeIcon icon={faPaperPlane} className="text-sm" />
        </button>
      </div>
    </div>
  )
}
