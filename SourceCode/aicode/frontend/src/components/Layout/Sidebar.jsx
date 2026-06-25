import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { useSocket } from '../../context/SocketContext'
import {
  LayoutDashboard, Scan, History, User, ShieldCheck, LogOut,
  Leaf, Wifi, WifiOff
} from 'lucide-react'

const navItems = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/recognition', icon: Scan,            label: 'Recognition' },
  { to: '/history',     icon: History,         label: 'History' },
  { to: '/profile',     icon: User,            label: 'Profile' },
]

export default function Sidebar() {
  const { user, logout } = useAuth()
  const { connected }    = useSocket()
  const navigate         = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col border-r border-white/10"
           style={{ background: 'rgba(10,15,30,0.95)', backdropFilter: 'blur(20px)' }}>

      {/* Logo */}
      <div className="px-6 py-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-primary-500/30">
            <Leaf className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display font-bold text-white text-lg leading-none">ThanhDinh</h1>
            <p className="text-xs text-slate-500 mt-0.5">Recognition System</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <Icon className="w-4.5 h-4.5 flex-shrink-0" size={18} />
            <span>{label}</span>
          </NavLink>
        ))}

        {user?.role === 'admin' && (
          <NavLink
            to="/admin"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <ShieldCheck size={18} />
            <span>Admin Panel</span>
          </NavLink>
        )}
      </nav>

      {/* User + status */}
      <div className="px-3 pb-4 space-y-2 border-t border-white/10 pt-4">
        {/* WS Status */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg">
          {connected
            ? <><Wifi size={14} className="text-primary-400" /><span className="text-xs text-primary-400">Live updates on</span></>
            : <><WifiOff size={14} className="text-slate-500" /><span className="text-xs text-slate-500">Connecting…</span></>
          }
        </div>

        {/* User info */}
        <div className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center text-sm font-bold text-white">
            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">{user?.name}</p>
            <p className="text-xs text-slate-500 truncate">{user?.role}</p>
          </div>
        </div>

        {/* Logout */}
        <button
          onClick={handleLogout}
          className="nav-item w-full text-red-400 hover:text-red-300 hover:bg-red-500/10"
        >
          <LogOut size={18} />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  )
}
