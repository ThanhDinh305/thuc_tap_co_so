import { useState } from 'react'
import { predictApi } from '../../services/api'
import { toast } from 'react-hot-toast'
import ImageUpload from './ImageUpload'
import WebcamCapture from './WebcamCapture'
import ResultCard from './ResultCard'
import { Upload, Camera } from 'lucide-react'

export default function Recognition() {
  const [mode,    setMode]    = useState('upload')   // 'upload' | 'webcam'
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)

  const handleUpload = async file => {
    setLoading(true)
    setResult(null)
    const previewUrl = URL.createObjectURL(file)
    setPreview(previewUrl)

    try {
      const form = new FormData()
      form.append('image', file)
      const res = await predictApi.upload(form)
      setResult(res.data.prediction)
      toast.success(`Detected: ${capitalize(res.data.prediction.fruit_name)} 🎯`)
    } catch (err) {
      const msg = err.response?.data?.message || 'Prediction failed.'
      toast.error(msg)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleWebcam = async base64 => {
    setLoading(true)
    setResult(null)
    setPreview(base64)
    try {
      const res = await predictApi.webcam({ image_base64: base64 })
      setResult(res.data.prediction)
      toast.success(`Detected: ${capitalize(res.data.prediction.fruit_name)} 📸`)
    } catch (err) {
      const msg = err.response?.data?.message || 'Prediction failed.'
      toast.error(msg)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setResult(null)
    setPreview(null)
    setLoading(false)
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Mode tabs */}
      <div className="flex items-center gap-2 glass-card p-1.5 w-fit">
        <button
          id="tab-upload"
          onClick={() => { setMode('upload'); reset() }}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200
            ${mode === 'upload' ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/30' : 'text-slate-400 hover:text-white'}`}
        >
          <Upload size={16} /> Upload Image
        </button>
        <button
          id="tab-webcam"
          onClick={() => { setMode('webcam'); reset() }}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200
            ${mode === 'webcam' ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/30' : 'text-slate-400 hover:text-white'}`}
        >
          <Camera size={16} /> Webcam
        </button>
      </div>

      <div className={`grid gap-6 ${result ? 'lg:grid-cols-2' : 'grid-cols-1 max-w-2xl'}`}>
        {/* Input panel */}
        <div className="space-y-4">
          {mode === 'upload'
            ? <ImageUpload onImage={handleUpload} loading={loading} preview={preview} onReset={reset} />
            : <WebcamCapture onCapture={handleWebcam} loading={loading} />
          }
        </div>

        {/* Result panel */}
        {result && (
          <div className="animate-slide-up">
            <ResultCard prediction={result} imagePreview={preview} />
          </div>
        )}

        {/* Loading state */}
        {loading && !result && (
          <div className="glass-card p-10 flex flex-col items-center justify-center gap-4 animate-fade-in">
            <div className="relative">
              <div className="w-16 h-16 spinner" />
              <div className="absolute inset-0 flex items-center justify-center text-2xl animate-pulse">🔍</div>
            </div>
            <div className="text-center">
              <p className="text-white font-medium">Analyzing image…</p>
              <p className="text-slate-400 text-sm mt-1">AI model is processing your fruit</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : '' }
