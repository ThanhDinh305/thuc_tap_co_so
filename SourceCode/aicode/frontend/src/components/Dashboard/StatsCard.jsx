export default function StatsCard({ title, value, icon: Icon, color = 'primary', subtitle, isText = false }) {
  const colorMap = {
    primary: { bg: 'from-primary-500/20 to-primary-600/10', border: 'border-primary-500/20', icon: 'text-primary-400', iconBg: 'bg-primary-500/20' },
    cyan:    { bg: 'from-cyan-500/20 to-cyan-600/10',       border: 'border-cyan-500/20',    icon: 'text-cyan-400',    iconBg: 'bg-cyan-500/20'    },
    violet:  { bg: 'from-violet-500/20 to-violet-600/10',   border: 'border-violet-500/20',  icon: 'text-violet-400',  iconBg: 'bg-violet-500/20'  },
    amber:   { bg: 'from-amber-500/20 to-amber-600/10',     border: 'border-amber-500/20',   icon: 'text-amber-400',   iconBg: 'bg-amber-500/20'   },
  }
  const c = colorMap[color] || colorMap.primary

  return (
    <div className={`glass-card p-5 bg-gradient-to-br ${c.bg} border ${c.border} transition-all duration-300 hover:scale-[1.02]`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</p>
          <p className={`mt-2 font-display font-bold text-white ${isText ? 'text-xl' : 'text-3xl'}`}>
            {value}
          </p>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`w-10 h-10 rounded-xl ${c.iconBg} flex items-center justify-center flex-shrink-0`}>
          <Icon className={c.icon} size={20} />
        </div>
      </div>
    </div>
  )
}
