import { useCallback, useState } from 'react'
import { Upload, ImagePlus, X, ZoomIn } from 'lucide-react'

export default function ImageUpload({ onImage, loading, preview, onReset }) {
  const [dragging, setDragging] = useState(false)

  const process = file => {
    if (!file) return
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
    if (!allowed.includes(file.type)) {
      alert('Please upload a JPG, PNG, WebP, or BMP image.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('Image must be under 10 MB.')
      return
    }
    onImage(file)
  }

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) process(file)
  }, [onImage])

  const handleFileChange = e => {
    const file = e.target.files[0]
    if (file) process(file)
    e.target.value = ''
  }

  if (preview) {
    return (
      <div className="glass-card p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-300">Image Preview</h3>
          <button onClick={onReset} className="text-slate-500 hover:text-red-400 transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="relative group rounded-xl overflow-hidden">
          <img
            src={preview}
            alt="Upload preview"
            className="w-full h-64 object-contain bg-dark-800 rounded-xl"
          />
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-xl">
            <ZoomIn className="text-white" size={32} />
          </div>
        </div>
        {!loading && (
          <label className="btn-secondary w-full justify-center cursor-pointer">
            <Upload size={16} /> Upload Different Image
            <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
          </label>
        )}
      </div>
    )
  }

  return (
    <div className="glass-card p-6 space-y-4">
      <div>
        <h3 className="section-title">Upload Fruit Image</h3>
        <p className="text-sm text-slate-400 mt-1">Drag & drop or click to select an image</p>
      </div>

      <label
        id="drop-zone"
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`drop-zone flex flex-col items-center justify-center gap-4 py-16 cursor-pointer select-none
          ${dragging ? 'active' : ''} ${loading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <div className="w-16 h-16 rounded-2xl bg-primary-500/15 flex items-center justify-center">
          <ImagePlus className="text-primary-400" size={32} />
        </div>
        <div className="text-center">
          <p className="text-white font-medium">Drop your image here</p>
          <p className="text-slate-500 text-sm mt-1">or click to browse</p>
          <p className="text-slate-600 text-xs mt-2">JPG, PNG, WebP, BMP • Max 10 MB</p>
        </div>
        <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} disabled={loading} />
      </label>

      {/* Supported fruits */}
      <div className="border-t border-white/10 pt-4">
        <p className="text-xs text-slate-500 mb-3 font-medium uppercase tracking-wider">Supported Fruits (10 Classes)</p>
        <div className="flex flex-wrap gap-2">
          {[
            ['🍎','Apple'], ['🥑','Avocado'], ['🍌','Banana'], ['🐉','Dragon Fruit'],
            ['🍋','Lemon'], ['🥭','Mango'], ['🍊','Orange'], ['🍈','Papaya'],
            ['🍍','Pineapple'], ['🍓','Strawberry'],
          ].map(([emoji, name]) => (
            <span key={name} className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-dark-700 border border-white/10 text-xs text-slate-400">
              <span>{emoji}</span> {name}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
