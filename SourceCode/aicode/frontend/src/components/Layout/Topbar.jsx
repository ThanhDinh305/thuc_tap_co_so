import { useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { Bell, Search } from 'lucide-react'
import { format } from 'date-fns'

const PAGE_TITLES = {
  '/dashboard':   { title: 'Dashboard',       subtitle: 'Your fruit recognition overview' },
  '/recognition': { title: 'Recognition',     subtitle: 'Upload or capture a fruit image' },
  '/history':     { title: 'History',         subtitle: 'View your past predictions' },
  '/profile':     { title: 'Profile',         subtitle: 'Manage your account settings' },
  '/admin':       { title: 'Admin Panel',     subtitle: 'System management' },
}

export default function Topbar() {
  const { pathname } = useLocation()
  const { user }     = useAuth()
  const page         = PAGE_TITLES[pathname] || { title: 'ThanhDinh', subtitle: '' }
  const now          = format(new Date(), 'EEEE, MMMM d, yyyy')

  return (
    <header className="flex items-center justify-between px-4 md:px-6 py-3 md:py-4 border-b border-white/10"
            style={{ background: 'rgba(10,15,30,0.8)', backdropFilter: 'blur(10px)' }}>
      <div className="pr-2 min-w-0 flex-1">
        <h2 className="text-lg md:text-xl font-bold text-white font-display truncate">{page.title}</h2>
        <p className="text-[10px] md:text-xs text-slate-500 mt-0.5 truncate">{page.subtitle}</p>
      </div>

      <div className="flex items-center gap-2 md:gap-3 flex-shrink-0">
        <div className="hidden lg:flex items-center gap-2 text-xs text-slate-500 border border-white/10 px-3 py-1.5 rounded-lg">
          <span>{now}</span>
        </div>

        <div className="relative">
          <span className="w-2 h-2 bg-primary-500 rounded-full absolute -top-0.5 -right-0.5 animate-pulse" />
          <button className="w-8 h-8 md:w-9 md:h-9 rounded-xl border border-white/10 flex items-center justify-center text-slate-400 hover:text-white hover:border-primary-500/50 transition-all">
            <Bell size={16} />
          </button>
        </div>

        <div className="flex items-center gap-2 border border-white/10 rounded-xl px-2 py-1.5 md:px-3">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center text-xs font-bold text-white">
            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <span className="text-sm text-white font-medium hidden sm:block">{user?.name}</span>
        </div>
      </div>
    </header>
  )
}
