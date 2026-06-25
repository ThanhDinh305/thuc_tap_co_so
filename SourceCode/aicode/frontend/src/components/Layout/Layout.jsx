import { Outlet, NavLink } from 'react-router-dom'
import Sidebar from './Sidebar'
import Topbar from './Topbar'
import { LayoutDashboard, Scan, History, User } from 'lucide-react'

const navItems = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dash' },
  { to: '/recognition', icon: Scan,            label: 'Scan' },
  { to: '/history',     icon: History,         label: 'History' },
  { to: '/profile',     icon: User,            label: 'Profile' },
]

export default function Layout() {
  return (
    <div className="flex h-[100dvh] overflow-hidden bg-dark-900">
      <div className="hidden md:flex h-full">
        <Sidebar />
      </div>
      <div className="flex flex-col flex-1 overflow-hidden relative">
        <Topbar />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 pb-24 md:pb-6">
          <Outlet />
        </main>

        {/* Mobile Bottom Nav */}
        <nav className="md:hidden absolute bottom-0 left-0 right-0 bg-[#0a0f1e]/95 backdrop-blur-xl border-t border-white/10 flex justify-around p-2 pb-safe z-50">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex flex-col items-center p-2 rounded-xl text-xs transition-colors ${
                  isActive ? 'text-primary-400 font-medium' : 'text-slate-400 hover:text-white'
                }`
              }
            >
              <Icon size={20} className="mb-1" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  )
}
