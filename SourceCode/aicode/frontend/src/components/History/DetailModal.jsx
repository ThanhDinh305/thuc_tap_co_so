import { X } from 'lucide-react'
import { format } from 'date-fns'

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}
const RIPENESS_LABEL = {
  ripe: 'Đã chín ✅', unripe: 'Chưa chín 🟢', overripe: 'Quá chín 🟤', unknown: 'Không rõ'
}

export default function DetailModal({ record, onClose }) {
  if (!record) return null
  const vitamins = typeof record.vitamins === 'string' ? JSON.parse(record.vitamins) : record.vitamins || {}

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
         onClick={onClose}>
      <div className="glass-card w-full max-w-2xl max-h-[90vh] overflow-y-auto animate-slide-up"
           onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <span className="text-4xl">{FRUIT_EMOJIS[record.fruit_name] || '🍑'}</span>
            <div>
              <h2 className="font-display text-xl font-bold text-white capitalize">{record.fruit_name}</h2>
              <p className="text-slate-400 text-sm">{format(new Date(record.created_at), 'PPpp')}</p>
            </div>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-lg text-slate-500 hover:text-white hover:bg-white/10 flex items-center justify-center transition-all">
            <X size={18} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Image */}
          {record.image_path && (
            <div className="rounded-xl overflow-hidden border border-white/10">
              <img src={record.image_path} alt={record.fruit_name} className="w-full h-52 object-contain bg-dark-800" />
            </div>
          )}

          {/* Key stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-primary-500/10 border border-primary-500/20 rounded-xl p-3 text-center">
              <p className="text-xs text-slate-400">Confidence</p>
              <p className="text-xl font-bold text-primary-400 mt-1">{parseFloat(record.confidence).toFixed(1)}%</p>
            </div>
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 text-center">
              <p className="text-xs text-slate-400">Calories</p>
              <p className="text-xl font-bold text-amber-400 mt-1">{record.calories ?? '—'} <span className="text-xs">kcal</span></p>
            </div>
            <div className="bg-violet-500/10 border border-violet-500/20 rounded-xl p-3 text-center">
              <p className="text-xs text-slate-400">Ripeness</p>
              <p className="text-sm font-bold text-violet-400 mt-1">{RIPENESS_LABEL[record.ripeness] || '—'}</p>
            </div>
          </div>

          {/* Macros */}
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Macronutrients (per 100g)</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Protein',       value: record.protein,       unit: 'g',   color: 'text-cyan-400'   },
                { label: 'Carbohydrates', value: record.carbohydrates, unit: 'g',   color: 'text-amber-400'  },
                { label: 'Fat',           value: record.fat,           unit: 'g',   color: 'text-violet-400' },
              ].map(({ label, value, unit, color }) => (
                <div key={label} className="bg-dark-700/60 rounded-lg p-3 flex justify-between items-center">
                  <span className="text-slate-400 text-sm">{label}</span>
                  <span className={`font-semibold ${color}`}>{value ?? '—'} <span className="text-xs text-slate-500">{unit}</span></span>
                </div>
              ))}
            </div>
          </div>

          {/* Vitamins */}
          {Object.keys(vitamins).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Vitamins & Minerals</h3>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(vitamins).filter(([,v]) => v != null).map(([k, v]) => (
                  <div key={k} className="bg-dark-700/60 rounded-lg px-3 py-2 text-center">
                    <p className="text-xs text-slate-500">{k.replace(/_/g,' ').replace('mg','').replace('iu','').trim()}</p>
                    <p className="text-sm font-semibold text-slate-200 mt-0.5">{v} <span className="text-xs text-slate-500">{k.includes('iu') ? 'IU' : 'mg'}</span></p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
