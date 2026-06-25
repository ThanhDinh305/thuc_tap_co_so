import { useState, useEffect } from 'react'
import { adminApi } from '../../services/api'
import { toast } from 'react-hot-toast'
import { Users, BarChart3, Trash2, Search, Shield, Activity } from 'lucide-react'
import { format } from 'date-fns'

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}

export default function AdminPanel() {
  const [tab,       setTab]       = useState('stats')    // 'stats' | 'users' | 'records'
  const [stats,     setStats]     = useState(null)
  const [users,     setUsers]     = useState([])
  const [records,   setRecords]   = useState([])
  const [loading,   setLoading]   = useState(false)
  const [search,    setSearch]    = useState('')
  const [userPage,  setUserPage]  = useState(1)
  const [recPage,   setRecPage]   = useState(1)
  const [userPagi,  setUserPagi]  = useState({})
  const [recPagi,   setRecPagi]   = useState({})

  useEffect(() => { fetchStats() }, [])
  useEffect(() => { if (tab === 'users')   fetchUsers(1)   }, [tab])
  useEffect(() => { if (tab === 'records') fetchRecords(1) }, [tab])

  const fetchStats = async () => {
    try {
      const res = await adminApi.getStats()
      setStats(res.data.data)
    } catch { toast.error('Failed to load stats.') }
  }

  const fetchUsers = async (pg = 1) => {
    setLoading(true)
    try {
      const res = await adminApi.getUsers({ page: pg, limit: 15, search })
      setUsers(res.data.data)
      setUserPagi(res.data.pagination)
      setUserPage(pg)
    } catch { toast.error('Failed to load users.') }
    finally { setLoading(false) }
  }

  const fetchRecords = async (pg = 1) => {
    setLoading(true)
    try {
      const res = await adminApi.getRecords({ page: pg, limit: 15, search })
      setRecords(res.data.data)
      setRecPagi(res.data.pagination)
      setRecPage(pg)
    } catch { toast.error('Failed to load records.') }
    finally { setLoading(false) }
  }

  const deleteUser = async id => {
    if (!window.confirm('Delete this user and all their data?')) return
    try {
      await adminApi.deleteUser(id)
      setUsers(prev => prev.filter(u => u.id !== id))
      toast.success('User deleted.')
    } catch (err) { toast.error(err.response?.data?.message || 'Failed.') }
  }

  const deleteRecord = async id => {
    if (!window.confirm('Delete this record?')) return
    try {
      await adminApi.deleteRecord(id)
      setRecords(prev => prev.filter(r => r.id !== id))
      toast.success('Record deleted.')
    } catch { toast.error('Failed.') }
  }

  const tabs = [
    { id: 'stats',   label: 'Overview',    icon: BarChart3 },
    { id: 'users',   label: 'Users',       icon: Users     },
    { id: 'records', label: 'All Records', icon: Activity  },
  ]

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Admin badge */}
      <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-violet-500/10 border border-violet-500/20 w-fit">
        <Shield size={16} className="text-violet-400" />
        <span className="text-sm font-semibold text-violet-400">Administrator Access</span>
      </div>

      {/* Tab navigation */}
      <div className="glass-card p-1.5 flex gap-1 w-fit">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            id={`admin-tab-${id}`}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all
              ${tab === id ? 'bg-violet-500 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}
          >
            <Icon size={15} /> {label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab === 'stats' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Total Users',       value: stats.totalUsers,       color: 'text-primary-400' },
              { label: 'Total Predictions', value: stats.totalPredictions, color: 'text-cyan-400'    },
              { label: 'Today',             value: stats.today,            color: 'text-amber-400'   },
              { label: 'This Week',         value: stats.thisWeek,         color: 'text-violet-400'  },
            ].map(({ label, value, color }) => (
              <div key={label} className="glass-card p-5 text-center">
                <p className={`text-3xl font-bold font-display ${color}`}>{value}</p>
                <p className="text-xs text-slate-400 mt-1 uppercase tracking-wider">{label}</p>
              </div>
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-4">
            {/* Fruit Distribution */}
            <div className="glass-card p-5">
              <h3 className="section-title mb-4">Fruit Distribution</h3>
              <div className="space-y-2">
                {stats.fruitDist?.map((item, i) => {
                  const total = stats.totalPredictions || 1
                  const pct = Math.round((item.count / total) * 100)
                  return (
                    <div key={item.fruit_name} className="flex items-center gap-3">
                      <span className="w-6 text-center">{FRUIT_EMOJIS[item.fruit_name] || '🍑'}</span>
                      <span className="text-sm text-slate-300 w-24 capitalize truncate">{item.fruit_name}</span>
                      <div className="flex-1 progress-bar h-2">
                        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="text-xs text-slate-400 w-10 text-right">{item.count}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Recent activity */}
            <div className="glass-card p-5">
              <h3 className="section-title mb-4">Recent Activity</h3>
              <div className="space-y-2">
                {stats.recentActivity?.map((a, i) => (
                  <div key={i} className="flex items-center gap-3 py-2 border-b border-white/5 last:border-0">
                    <span>{FRUIT_EMOJIS[a.fruit_name] || '🍑'}</span>
                    <div className="flex-1 min-w-0">
                      <span className="text-sm text-white capitalize">{a.fruit_name}</span>
                      <span className="text-xs text-slate-500 ml-2">{a.user_name}</span>
                    </div>
                    <span className="text-xs text-primary-400">{parseFloat(a.confidence).toFixed(0)}%</span>
                    <span className="text-xs text-slate-500">{format(new Date(a.created_at), 'MMM d')}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Users */}
      {tab === 'users' && (
        <div className="space-y-3">
          <div className="flex gap-3">
            <div className="relative flex-1 max-w-sm">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search users…"
                     className="input-field pl-9 py-2 text-sm" />
            </div>
            <button onClick={() => fetchUsers(1)} className="btn-primary py-2">Search</button>
          </div>

          <div className="glass-card overflow-hidden">
            <table className="data-table">
              <thead>
                <tr><th>#</th><th>Name</th><th>Email</th><th>Role</th><th>Predictions</th><th>Joined</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={7} className="text-center py-8 text-slate-400">Loading…</td></tr>
                ) : users.map(u => (
                  <tr key={u.id}>
                    <td className="text-slate-500 text-xs">{u.id}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center text-xs font-bold text-white">
                          {u.name.charAt(0)}
                        </div>
                        <span className="font-medium text-white">{u.name}</span>
                      </div>
                    </td>
                    <td className="text-slate-400 text-xs">{u.email}</td>
                    <td>
                      <span className={u.role === 'admin' ? 'badge badge-purple' : 'badge badge-green'}>
                        {u.role}
                      </span>
                    </td>
                    <td className="text-center text-slate-300">{u.prediction_count}</td>
                    <td className="text-slate-500 text-xs">{format(new Date(u.created_at), 'MMM d, yyyy')}</td>
                    <td>
                      {u.role !== 'admin' && (
                        <button onClick={() => deleteUser(u.id)} className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all">
                          <Trash2 size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {userPagi.totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button disabled={userPage <= 1} onClick={() => fetchUsers(userPage - 1)} className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30">← Prev</button>
              <span className="text-sm text-slate-400">Page {userPage} of {userPagi.totalPages}</span>
              <button disabled={userPage >= userPagi.totalPages} onClick={() => fetchUsers(userPage + 1)} className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30">Next →</button>
            </div>
          )}
        </div>
      )}

      {/* Records */}
      {tab === 'records' && (
        <div className="space-y-3">
          <div className="flex gap-3">
            <div className="relative flex-1 max-w-sm">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search fruit or user…"
                     className="input-field pl-9 py-2 text-sm" />
            </div>
            <button onClick={() => fetchRecords(1)} className="btn-primary py-2">Search</button>
          </div>

          <div className="glass-card overflow-hidden">
            <table className="data-table">
              <thead>
                <tr><th>Fruit</th><th>User</th><th>Confidence</th><th>Calories</th><th>Date</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={6} className="text-center py-8 text-slate-400">Loading…</td></tr>
                ) : records.map(r => (
                  <tr key={r.id}>
                    <td>
                      <div className="flex items-center gap-2">
                        <span>{FRUIT_EMOJIS[r.fruit_name] || '🍑'}</span>
                        <span className="text-white capitalize font-medium">{r.fruit_name}</span>
                      </div>
                    </td>
                    <td className="text-slate-400 text-xs">
                      <div>{r.user_name}</div>
                      <div className="text-slate-600">{r.user_email}</div>
                    </td>
                    <td><span className="text-primary-400 font-semibold">{parseFloat(r.confidence).toFixed(1)}%</span></td>
                    <td className="text-amber-400">{r.calories ? `${r.calories} kcal` : '—'}</td>
                    <td className="text-slate-500 text-xs">{format(new Date(r.created_at), 'MMM d, yyyy HH:mm')}</td>
                    <td>
                      <button onClick={() => deleteRecord(r.id)} className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all">
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {recPagi.totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button disabled={recPage <= 1} onClick={() => fetchRecords(recPage - 1)} className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30">← Prev</button>
              <span className="text-sm text-slate-400">Page {recPage} of {recPagi.totalPages}</span>
              <button disabled={recPage >= recPagi.totalPages} onClick={() => fetchRecords(recPage + 1)} className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30">Next →</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
