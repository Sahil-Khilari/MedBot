import { useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCheck, faXmark, faPaperPlane } from '@fortawesome/free-solid-svg-icons'

export default function ValidationCard({ onValidate }) {
  const [selected, setSelected] = useState(null) // 'yes' | 'no'
  const [comments, setComments] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const submit = () => {
    if (!selected || submitted) return
    setSubmitted(true)
    onValidate(selected, comments)
  }

  if (submitted) {
    return (
      <div className="mt-2 px-3 py-2 bg-slate-50 rounded-xl text-xs text-slate-500 border border-slate-200">
        Validation submitted — processing...
      </div>
    )
  }

  return (
    <div className="mt-3 border border-amber-200 bg-amber-50 rounded-xl p-3">
      <p className="text-xs font-semibold text-amber-800 mb-3">
        ⚕ Human Validation Required — Do you agree with this AI result?
      </p>

      <div className="flex gap-2 mb-3">
        <button
          onClick={() => setSelected('yes')}
          className={`flex-1 flex flex-col items-center justify-center gap-1 py-3 rounded-xl border-2 text-sm font-medium transition-all ${
            selected === 'yes'
              ? 'border-green-500 bg-green-50 text-green-700'
              : 'border-slate-200 bg-white text-slate-600 hover:border-green-300'
          }`}
        >
          <FontAwesomeIcon icon={faCheck} className="text-lg" />
          Yes, I Agree
        </button>
        <button
          onClick={() => setSelected('no')}
          className={`flex-1 flex flex-col items-center justify-center gap-1 py-3 rounded-xl border-2 text-sm font-medium transition-all ${
            selected === 'no'
              ? 'border-red-500 bg-red-50 text-red-700'
              : 'border-slate-200 bg-white text-slate-600 hover:border-red-300'
          }`}
        >
          <FontAwesomeIcon icon={faXmark} className="text-lg" />
          No, Review Needed
        </button>
      </div>

      {selected && (
        <div className="space-y-2">
          <textarea
            value={comments}
            onChange={e => setComments(e.target.value)}
            placeholder="Add clinical comments (optional)..."
            rows={2}
            className="w-full text-xs border border-slate-200 rounded-lg px-3 py-2 resize-none focus:outline-none focus:border-sky-400"
          />
          <button
            onClick={submit}
            className="w-full flex items-center justify-center gap-2 py-2 bg-sky-500 hover:bg-sky-600 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <FontAwesomeIcon icon={faPaperPlane} />
            Submit Validation
          </button>
        </div>
      )}
    </div>
  )
}
