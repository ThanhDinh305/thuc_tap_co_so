import { CheckCircle, Zap } from 'lucide-react'

const FRUIT_EMOJIS = {
  apple: '🍎', avocado: '🥑', banana: '🍌', 'dragon fruit': '🐉',
  lemon: '🍋', mango: '🥭', orange: '🍊', papaya: '🍈',
  pineapple: '🍍', strawberry: '🍓',
}

const RIPENESS_CONFIG = {
  ripe:     { label: 'Đã chín ✅',    badge: 'badge-green',  bar: 'bg-primary-500' },
  unripe:   { label: 'Chưa chín 🟢',  badge: 'badge-blue',   bar: 'bg-cyan-500'    },
  overripe: { label: 'Quá chín 🟤',   badge: 'badge-yellow', bar: 'bg-amber-500'   },
  unknown:  { label: 'Không rõ',       badge: 'badge-purple', bar: 'bg-violet-500'  },
}

function NutritionBar({ label, value, unit, max, color = 'bg-primary-500' }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300 font-medium">{value} {unit}</span>
      </div>
      <div className="progress-bar">
        <div className={`progress-bar-fill ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function ResultCard({ prediction, imagePreview }) {
  if (!prediction) return null

  // Backend trả về record từ DB, nutrition nằm trong các cột riêng biệt
  // Đồng thời hỗ trợ cả cấu trúc trực tiếp từ AI Service (có object nutrition)
  let n = {}
  if (prediction.nutrition && typeof prediction.nutrition === 'object') {
    // Cấu trúc từ AI Service trực tiếp
    n = prediction.nutrition
  } else {
    // Cấu trúc từ DB record
    const vitamins = typeof prediction.vitamins === 'string'
      ? (() => { try { return JSON.parse(prediction.vitamins) } catch { return {} } })()
      : (prediction.vitamins || {})
    n = {
      energy_kcal:  prediction.calories,
      protein_g:    prediction.protein,
      fat_g:        prediction.fat,
      carbs_g:      prediction.carbohydrates,
      fiber_g:      vitamins.fiber_g,
      sugar_g:      vitamins.sugar_g,
      vitamin_c_mg: vitamins.vitamin_c_mg,
      vitamin_a_iu: vitamins.vitamin_a_iu,
      calcium_mg:   vitamins.calcium_mg,
      iron_mg:      vitamins.iron_mg,
      potassium_mg: vitamins.potassium_mg,
      magnesium_mg: vitamins.magnesium_mg,
    }
  }

  const r    = RIPENESS_CONFIG[prediction.ripeness] || RIPENESS_CONFIG.unknown
  const conf = parseFloat(prediction.confidence)
  const emoji = FRUIT_EMOJIS[prediction.fruit_name] || '🍑'

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-white/10"
           style={{ background: 'linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.08))' }}>
        <div className="flex items-start gap-4">
          {imagePreview && (
            <img src={imagePreview} alt="Fruit" className="w-20 h-20 object-cover rounded-xl border border-white/10 flex-shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle className="text-primary-400 flex-shrink-0" size={18} />
              <span className="text-xs text-primary-400 font-semibold uppercase tracking-wider">Detected</span>
            </div>
            <h2 className="font-display text-2xl font-bold text-white">
              {emoji} {capitalize(prediction.fruit_name)}
            </h2>
            {n.name_vn && <p className="text-slate-400 text-sm">{n.name_vn}</p>}
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              <span className={r.badge}>{r.label}</span>
            </div>
          </div>
        </div>

        {/* Confidence */}
        <div className="mt-4">
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-slate-400 flex items-center gap-1"><Zap size={12} />Confidence</span>
            <span className="font-bold text-primary-400">{conf.toFixed(1)}%</span>
          </div>
          <div className="progress-bar h-3">
            <div className="progress-bar-fill" style={{ width: `${conf}%` }} />
          </div>
        </div>
      </div>

      {/* Nutrition */}
      {n.energy_kcal != null && (
        <div className="p-5 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Nutrition per 100g</h3>

            {/* Calories highlight */}
            <div className="flex items-center justify-center py-3 mb-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
              <span className="text-3xl font-bold text-amber-400">{n.energy_kcal}</span>
              <span className="ml-2 text-sm text-amber-300">kcal</span>
            </div>

            <div className="space-y-3">
              <NutritionBar label="Protein"       value={n.protein_g}  unit="g"  max={30}  color="bg-cyan-500"    />
              <NutritionBar label="Carbohydrates" value={n.carbs_g}    unit="g"  max={50}  color="bg-amber-500"   />
              <NutritionBar label="Fat"           value={n.fat_g}      unit="g"  max={20}  color="bg-violet-500"  />
              <NutritionBar label="Fiber"         value={n.fiber_g}    unit="g"  max={15}  color="bg-primary-500" />
              <NutritionBar label="Sugar"         value={n.sugar_g}    unit="g"  max={30}  color="bg-pink-500"    />
            </div>
          </div>

          {/* Vitamins & Minerals */}
          <div className="border-t border-white/10 pt-4">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Vitamins & Minerals</h4>
            <div className="grid grid-cols-2 gap-2">
              {[
                { k: 'vitamin_c_mg',  label: 'Vitamin C', unit: 'mg' },
                { k: 'vitamin_a_iu',  label: 'Vitamin A', unit: 'IU' },
                { k: 'calcium_mg',    label: 'Calcium',   unit: 'mg' },
                { k: 'iron_mg',       label: 'Iron',      unit: 'mg' },
                { k: 'potassium_mg',  label: 'Potassium', unit: 'mg' },
                { k: 'magnesium_mg',  label: 'Magnesium', unit: 'mg' },
              ].map(({ k, label, unit }) => (
                n[k] != null && (
                  <div key={k} className="bg-dark-700/60 rounded-lg px-3 py-2">
                    <p className="text-xs text-slate-500">{label}</p>
                    <p className="text-sm font-semibold text-slate-200">{n[k]} <span className="text-xs text-slate-500">{unit}</span></p>
                  </div>
                )
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : '' }
