export default function ThinkingIndicator() {
  return (
    <div className="flex gap-3 animate-slideIn px-2">
      <div className="w-8 h-8 rounded-full bg-sky-100 flex items-center justify-center flex-shrink-0 relative">
        <span className="text-xs text-sky-600 font-bold">AI</span>
        <span className="absolute inset-0 rounded-full border-2 border-sky-300 animate-pulse-ring" />
      </div>
      <div className="flex items-center gap-2 bg-white rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-center">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
        <span className="text-xs text-slate-400 animate-pulse ml-1">Analyzing...</span>
      </div>
    </div>
  )
}
