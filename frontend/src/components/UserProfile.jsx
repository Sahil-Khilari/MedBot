import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faRightFromBracket } from '@fortawesome/free-solid-svg-icons'

export default function UserProfile({ user, onLogout }) {
  if (!user) return null

  return (
    <div className="flex items-center gap-3 px-3 py-3 border-t border-sidebar-border">
      {user.picture ? (
        <img src={user.picture} alt={user.name} className="w-8 h-8 rounded-full object-cover flex-shrink-0" />
      ) : (
        <div className="w-8 h-8 rounded-full bg-sky-500 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
          {user.name?.[0]?.toUpperCase() || 'U'}
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-slate-200 truncate">{user.name}</p>
        <p className="text-[10px] text-slate-500 truncate">{user.email}</p>
      </div>
      <button
        onClick={onLogout}
        className="text-slate-500 hover:text-red-400 transition-colors p-1"
        title="Sign out"
      >
        <FontAwesomeIcon icon={faRightFromBracket} className="text-sm" />
      </button>
    </div>
  )
}
