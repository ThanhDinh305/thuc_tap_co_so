import { useState, useEffect, useCallback } from 'react'
import { historyApi } from '../../services/api'
import { useSocket } from '../../context/SocketContext'
import { toast } from 'react-hot-toast'
import { Search, CalendarDays, Trash2, FileSpreadsheet, FileText, RefreshCw } from 'lucide-react'
import DetailModal from './DetailModal'
import { format } from 'date-fns'

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}
const RIPENESS_BADGE = {
  ripe:     'badge-green',
  unripe:   'badge-blue',
  overripe: 'badge-yellow',
  unknown:  'badge-purple',
}
const RIPENESS_LABEL = {
  ripe: 'Đã chín', unripe: 'Chưa chín', overripe: 'Quá chín', unknown: 'N/A'
}

export default function History() {
  const [records,   setRecords]   = useState([])
  const [loading,   setLoading]   = useState(true)
  const [search,    setSearch]    = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate,   setEndDate]   = useState('')
  const [page,      setPage]      = useState(1)
  const [pagination,setPagination]= useState({ total: 0, totalPages: 1 })
  const [selected,  setSelected]  = useState(null)   // for detail modal
  const { subscribe } = useSocket()

  const fetchHistory = useCallback(async (pg = 1) => {
    setLoading(true)
    try {
      const res = await historyApi.getAll({ page: pg, limit: 10, search, startDate, endDate })
      setRecords(res.data.data)
      setPagination(res.data.pagination)
      setPage(pg)
    } catch { toast.error('Failed to load history.') }
    finally { setLoading(false) }
  }, [search, startDate, endDate])

  // Initial load
  useEffect(() => { fetchHistory(1) }, [fetchHistory])

  // Real-time: prepend new records from WebSocket
  useEffect(() => {
    const unsub = subscribe('new_prediction', record => {
      setRecords(prev => [record, ...prev])
      setPagination(p => ({ ...p, total: p.total + 1 }))
      toast.success(`New prediction saved: ${capitalize(record.fruit_name)} 🍎`, { duration: 3000 })
    })
    return unsub
  }, [subscribe])

  const handleDelete = async id => {
    if (!window.confirm('Delete this record?')) return
    try {
      await historyApi.deleteOne(id)
      setRecords(prev => prev.filter(r => r.id !== id))
      toast.success('Record deleted.')
    } catch { toast.error('Failed to delete.') }
  }

  const handleDeleteAll = async () => {
    if (!window.confirm('Delete ALL history? This cannot be undone.')) return
    try {
      await historyApi.deleteAll()
      setRecords([])
      setPagination({ total: 0, totalPages: 1 })
      toast.success('All history cleared.')
    } catch { toast.error('Failed to delete history.') }
  }

  const downloadBlob = (blob, filename) => {
    const url = URL.createObjectURL(blob)
    const a   = document.createElement('a')
    a.href = url; a.download = filename; a.click()
    URL.revokeObjectURL(url)
  }

  const exportExcel = async () => {
    try {
      const res = await historyApi.exportExcel()
      downloadBlob(res.data, 'fruit_history.xlsx')
      toast.success('Excel exported!')
    } catch { toast.error('Export failed.') }
  }

  const exportPDF = async () => {
    try {
      const res = await historyApi.exportPDF()
      downloadBlob(res.data, 'fruit_history.pdf')
      toast.success('PDF exported!')
    } catch { toast.error('PDF export failed.') }
  }

  const handleSearch = e => { e.preventDefault(); fetchHistory(1) }

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Toolbar */}
      <div className="glass-card p-4">
        <form onSubmit={handleSearch} className="flex flex-wrap gap-3 items-end">
          {/* Search */}
          <div className="relative flex-1 min-w-48">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              id="history-search"
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search fruit name…"
              className="input-field pl-9 py-2 text-sm"
            />
          </div>

          {/* Date range */}
          <div className="flex items-center gap-2">
            <CalendarDays size={15} className="text-slate-500 flex-shrink-0" />
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="input-field py-2 text-sm w-36"
            />
            <span className="text-slate-500 text-sm">to</span>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="input-field py-2 text-sm w-36"
            />
          </div>

          <button type="submit" className="btn-primary py-2">
            <Search size={15} /> Search
          </button>
          <button type="button" onClick={() => { setSearch(''); setStartDate(''); setEndDate(''); }} className="btn-secondary py-2">
            <RefreshCw size={15} /> Reset
          </button>
        </form>
      </div>

      {/* Action buttons */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-400">
          <span className="text-white font-semibold">{pagination.total}</span> records found
        </div>
        <div className="flex gap-2">
          <button onClick={exportExcel} className="btn-secondary py-1.5 px-3 text-xs">
            <FileSpreadsheet size={14} /> Excel
          </button>
          <button onClick={exportPDF} className="btn-secondary py-1.5 px-3 text-xs">
            <FileText size={14} /> PDF
          </button>
          {records.length > 0 && (
            <button onClick={handleDeleteAll} className="btn-danger py-1.5 px-3 text-xs">
              <Trash2 size={14} /> Clear All
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="p-12 text-center">
            <div className="spinner w-10 h-10 mx-auto mb-3" />
            <p className="text-slate-400 text-sm">Loading history…</p>
          </div>
        ) : records.length === 0 ? (
          <div className="p-12 text-center">
            <div className="text-5xl mb-3">📋</div>
            <p className="text-white font-medium">No records found</p>
            <p className="text-slate-400 text-sm mt-1">Try a different search or start recognizing fruits!</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Fruit</th>
                  <th>Confidence</th>
                  <th>Ripeness</th>
                  <th>Calories</th>
                  <th>Protein</th>
                  <th>Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {records.map(record => (
                  <tr key={record.id} className="cursor-pointer" onClick={() => setSelected(record)}>
                    <td onClick={e => e.stopPropagation()}>
                      <img
                        src={record.image_path}
                        alt={record.fruit_name}
                        className="w-12 h-12 object-cover rounded-lg border border-white/10"
                        onError={e => { e.target.style.display='none'; e.target.nextSibling.style.display='flex' }}
                      />
                      <div className="w-12 h-12 rounded-lg bg-dark-700 items-center justify-center text-xl hidden">
                        {FRUIT_EMOJIS[record.fruit_name] || '🍑'}
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span>{FRUIT_EMOJIS[record.fruit_name] || '🍑'}</span>
                        <span className="font-medium text-white capitalize">{record.fruit_name}</span>
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="progress-bar w-16 h-1.5">
                          <div className="progress-bar-fill" style={{ width: `${record.confidence}%` }} />
                        </div>
                        <span className="text-primary-400 text-xs font-semibold">{parseFloat(record.confidence).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={RIPENESS_BADGE[record.ripeness] || 'badge badge-purple'}>
                        {RIPENESS_LABEL[record.ripeness] || 'N/A'}
                      </span>
                    </td>
                    <td className="text-amber-400 font-medium">{record.calories ? `${record.calories} kcal` : '—'}</td>
                    <td>{record.protein ? `${record.protein}g` : '—'}</td>
                    <td className="text-slate-500 text-xs whitespace-nowrap">
                      {format(new Date(record.created_at), 'MMM d, yyyy HH:mm')}
                    </td>
                    <td onClick={e => e.stopPropagation()}>
                      <button
                        onClick={() => handleDelete(record.id)}
                        className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                        title="Delete"
                      >
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {pagination.totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button
            disabled={page <= 1}
            onClick={() => fetchHistory(page - 1)}
            className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="text-sm text-slate-400">
            Page <span className="text-white font-semibold">{page}</span> of {pagination.totalPages}
          </span>
          <button
            disabled={page >= pagination.totalPages}
            onClick={() => fetchHistory(page + 1)}
            className="btn-secondary py-1.5 px-4 text-sm disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}

      {/* Detail Modal */}
      {selected && <DetailModal record={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : '' }
