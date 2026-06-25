import { useRef, useState, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Camera, RefreshCw, ZapOff, Zap } from 'lucide-react'

export default function WebcamCapture({ onCapture, loading }) {
  const webcamRef = useRef(null)
  const [camActive,  setCamActive]  = useState(false)
  const [facingMode, setFacingMode] = useState('user')
  const [error,      setError]      = useState(null)

  const capture = useCallback(() => {
    if (!webcamRef.current) return
    const imageSrc = webcamRef.current.getScreenshot()
    if (imageSrc) onCapture(imageSrc)
  }, [onCapture])

  const toggleCamera = () => {
    setCamActive(prev => !prev)
    setError(null)
  }

  return (
    <div className="glass-card p-6 space-y-4">
      <div>
        <h3 className="section-title">Webcam Capture</h3>
        <p className="text-sm text-slate-400 mt-1">Use your camera to capture a fruit</p>
      </div>

      {/* Camera viewport */}
      <div className="relative rounded-xl overflow-hidden bg-dark-800 border border-white/10"
           style={{ aspectRatio: '16/10' }}>
        {camActive ? (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.95}
            videoConstraints={{ facingMode, width: 1280, height: 720 }}
            onUserMediaError={e => setError(e.message || 'Camera error')}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="w-20 h-20 rounded-full border-2 border-dashed border-slate-600 flex items-center justify-center">
              <Camera className="text-slate-500" size={32} />
            </div>
            <p className="text-slate-500 text-sm">Camera is off</p>
          </div>
        )}

        {/* Scanner overlay */}
        {camActive && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary-400 rounded-tl-lg" />
            <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary-400 rounded-tr-lg" />
            <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary-400 rounded-bl-lg" />
            <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary-400 rounded-br-lg" />
          </div>
        )}
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          ⚠ {error}. Please allow camera access in your browser.
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-3">
        <button
          id="btn-toggle-camera"
          onClick={toggleCamera}
          className={camActive ? 'btn-danger flex-1' : 'btn-primary flex-1'}
        >
          {camActive ? <><ZapOff size={16} /> Stop Camera</> : <><Zap size={16} /> Start Camera</>}
        </button>

        {camActive && (
          <>
            <button
              id="btn-flip-camera"
              onClick={() => setFacingMode(m => m === 'user' ? 'environment' : 'user')}
              className="btn-secondary px-3"
              title="Flip camera"
            >
              <RefreshCw size={16} />
            </button>

            <button
              id="btn-capture"
              onClick={capture}
              disabled={loading}
              className="btn-primary flex-1"
            >
              {loading ? <span className="spinner w-4 h-4" /> : <><Camera size={16} /> Capture & Analyze</>}
            </button>
          </>
        )}
      </div>

      {camActive && (
        <p className="text-xs text-slate-500 text-center">
          Hold a fruit up to the camera, then click <strong className="text-slate-400">Capture & Analyze</strong>
        </p>
      )}
    </div>
  )
}
