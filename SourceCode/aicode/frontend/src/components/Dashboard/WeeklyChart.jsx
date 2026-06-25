import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend, Filler
} from 'chart.js'
import { format, parseISO } from 'date-fns'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, Filler)

export default function WeeklyChart({ data = [] }) {
  const labels = data.map(d => {
    try { return format(parseISO(d.date), 'EEE\nMMM d') } catch { return d.date }
  })
  const counts = data.map(d => d.count)

  const chartData = {
    labels,
    datasets: [{
      label: 'Predictions',
      data: counts,
      backgroundColor: 'rgba(16,185,129,0.25)',
      borderColor:      'rgba(16,185,129,0.9)',
      borderWidth: 2,
      borderRadius: 8,
      borderSkipped: false,
      hoverBackgroundColor: 'rgba(16,185,129,0.45)',
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
        callbacks: {
          label: ctx => ` ${ctx.raw} prediction${ctx.raw !== 1 ? 's' : ''}`,
        }
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
        ticks: { color: '#64748b', font: { size: 11 } },
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#64748b', font: { size: 11 }, precision: 0 },
      },
    },
  }

  if (!data.length) return <div className="h-48 flex items-center justify-center text-slate-500 text-sm">No data yet</div>

  return <div style={{ height: 220 }}><Bar data={chartData} options={options} /></div>
}
