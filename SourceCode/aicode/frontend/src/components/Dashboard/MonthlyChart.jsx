import { Bar } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function MonthlyChart({ data = [] }) {
  const labels = data.map(d => {
    const [y, m] = d.month.split('-')
    const date = new Date(parseInt(y), parseInt(m) - 1)
    return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  })
  const counts = data.map(d => d.count)

  const chartData = {
    labels,
    datasets: [{
      label: 'Predictions',
      data: counts,
      backgroundColor: counts.map((_, i) => `hsla(${170 + i * 8}, 70%, 50%, 0.3)`),
      borderColor:     counts.map((_, i) => `hsla(${170 + i * 8}, 70%, 55%, 1)`),
      borderWidth: 2,
      borderRadius: 6,
      borderSkipped: false,
    }],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15,23,42,0.9)',
        borderColor: 'rgba(16,185,129,0.3)',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        padding: 10,
        cornerRadius: 8,
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#64748b', font: { size: 10 } },
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#64748b', font: { size: 10 }, precision: 0 },
      },
    },
  }

  if (!data.length) return <div className="h-48 flex items-center justify-center text-slate-500 text-sm">No data yet</div>

  return <div style={{ height: 240 }}><Bar data={chartData} options={options} /></div>
}
