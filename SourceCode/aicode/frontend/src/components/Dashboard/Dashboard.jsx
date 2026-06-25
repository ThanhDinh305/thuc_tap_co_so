import { useEffect, useState } from 'react'
import { dashboardApi } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import StatsCard from './StatsCard'
import WeeklyChart from './WeeklyChart'
import MonthlyChart from './MonthlyChart'
import FruitPieChart from './FruitPieChart'
import { Scan, CalendarDays, CalendarCheck, Trophy, TrendingUp } from 'lucide-react'

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}

export default function Dashboard() {
  const { user } = useAuth()
  const [stats,        setStats]        = useState(null)
  const [weeklyData,   setWeeklyData]   = useState([])
  const [monthlyData,  setMonthlyData]  = useState([])
  const [distribution, setDistribution] = useState([])
  const [loading,      setLoading]      = useState(true)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [s, w, m, d] = await Promise.all([
          dashboardApi.getStats(),
          dashboardApi.getWeekly(),
          dashboardApi.getMonthly(),
          dashboardApi.getDistribution(),
        ])
        setStats(s.data.data)
        setWeeklyData(w.data.data)
        setMonthlyData(m.data.data)
        setDistribution(d.data.data)
      } catch (err) {
        console.error('Dashboard fetch error:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center space-y-3">
        <div className="spinner w-12 h-12 mx-auto" />
        <p className="text-slate-400 text-sm">Loading dashboard…</p>
      </div>
    </div>
  )

  const topFruitEmoji = stats?.topFruit ? (FRUIT_EMOJIS[stats.topFruit.fruit_name] || '🍑') : '—'

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome banner */}
      <div className="glass-card p-6 relative overflow-hidden"
           style={{ background: 'linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.08) 50%, rgba(124,58,237,0.1) 100%)' }}>
        <div className="absolute right-6 top-4 text-6xl opacity-20 pointer-events-none fruit-bounce">🍎</div>
        <h2 className="font-display text-2xl font-bold text-white">
          Good {getGreeting()}, <span className="gradient-text">{user?.name?.split(' ')[0]}</span>! 👋
        </h2>
        <p className="text-slate-400 mt-1 text-sm">
          Here's what's happening with your fruit recognition activity.
        </p>
        <div className="flex items-center gap-2 mt-3">
          <div className="px-3 py-1 rounded-full bg-primary-500/20 border border-primary-500/30 text-primary-400 text-xs font-medium">
            🤖 AI Model: YOLOv8 Active
          </div>
          <div className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 text-xs font-medium">
            10 Fruit Classes
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Total Predictions"
          value={stats?.total ?? 0}
          icon={Scan}
          color="primary"
          trend={null}
        />
        <StatsCard
          title="Today"
          value={stats?.today ?? 0}
          icon={CalendarDays}
          color="cyan"
          subtitle="predictions today"
        />
        <StatsCard
          title="This Week"
          value={stats?.thisWeek ?? 0}
          icon={CalendarCheck}
          color="violet"
          subtitle="this week"
        />
        <StatsCard
          title="Top Fruit"
          value={stats?.topFruit ? `${topFruitEmoji} ${capitalize(stats.topFruit.fruit_name)}` : '—'}
          icon={Trophy}
          color="amber"
          subtitle={stats?.topFruit ? `${stats.topFruit.count} times` : 'no data yet'}
          isText
        />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="section-title">Weekly Activity</h3>
            <span className="badge badge-green">Last 7 Days</span>
          </div>
          <WeeklyChart data={weeklyData} />
        </div>

        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="section-title">Fruit Distribution</h3>
            <span className="badge badge-purple">All Time</span>
          </div>
          <FruitPieChart data={distribution} />
        </div>
      </div>

      {/* Monthly chart */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="section-title">Monthly Trend</h3>
            <p className="text-xs text-slate-500 mt-0.5">Predictions over the last 12 months</p>
          </div>
          <TrendingUp className="text-primary-400" size={20} />
        </div>
        <MonthlyChart data={monthlyData} />
      </div>
    </div>
  )
}

function getGreeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Morning'
  if (h < 17) return 'Afternoon'
  return 'Evening'
}
function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : '' }
