import { Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend)

const COLORS = [
  '#10b981','#06b6d4','#8b5cf6','#f59e0b','#ef4444',
  '#3b82f6','#ec4899','#14b8a6','#f97316','#84cc16',
]

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}

export default function FruitPieChart({ data = [] }) {
  if (!data.length) return (
    <div className="h-48 flex flex-col items-center justify-center text-slate-500 text-sm gap-2">
      <span className="text-3xl">🍽️</span>
      <span>No predictions yet</span>
    </div>
  )

  const labels  = data.map(d => d.fruit_name)
  const counts  = data.map(d => d.count)
  const colors  = labels.map((_, i) => COLORS[i % COLORS.length])

  const chartData = {
    labels,
    datasets: [{
      data:            counts,
      backgroundColor: colors.map(c => c + '80'),
      borderColor:     colors,
      borderWidth: 2,
      hoverOffset: 8,
    }],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '60%',
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15,23,42,0.9)',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: ctx => ` ${ctx.label}: ${ctx.raw} (${Math.round(ctx.parsed / counts.reduce((a,b) => a+b,0) * 100)}%)`,
        }
      },
    },
  }

  const total = counts.reduce((a, b) => a + b, 0)

  return (
    <div>
      <div className="relative" style={{ height: 180 }}>
        <Doughnut data={chartData} options={options} />
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-2xl font-bold text-white">{total}</span>
          <span className="text-xs text-slate-400">total</span>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 space-y-1.5 max-h-36 overflow-y-auto scrollbar-hide">
        {data.map((item, i) => (
          <div key={item.fruit_name} className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
              <span className="text-slate-400 capitalize">
                {FRUIT_EMOJIS[item.fruit_name] || '🍑'} {item.fruit_name}
              </span>
            </div>
            <span className="text-slate-300 font-medium">{item.count}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
