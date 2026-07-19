import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faXmark } from '@fortawesome/free-solid-svg-icons'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

export default function ImagePreviewStrip({ image, onRemove }) {
  if (!image) return null
  const name = image.file.name.length > 24 ? image.file.name.slice(0, 21) + '...' : image.file.name

  return (
    <div className="flex-shrink-0 border-t border-slate-200 bg-white px-4 py-2 flex items-center gap-3">
      <img src={image.dataUrl} alt="Preview" className="w-12 h-12 rounded-lg object-contain border border-slate-200 bg-slate-50" />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-slate-700 truncate">{name}</p>
        <p className="text-xs text-slate-400">{formatSize(image.file.size)}</p>
      </div>
      <button
        onClick={onRemove}
        className="text-slate-400 hover:text-red-500 transition-colors p-1"
        title="Remove image"
      >
        <FontAwesomeIcon icon={faXmark} />
      </button>
    </div>
  )
}
