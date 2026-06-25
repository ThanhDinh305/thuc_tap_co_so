import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { toast } from 'react-hot-toast'
import { Leaf, Eye, EyeOff, Mail, Lock, LogIn } from 'lucide-react'

export default function Login() {
  const [form, setForm]       = useState({ email: '', password: '' })
  const [show, setShow]       = useState(false)
  const [loading, setLoading] = useState(false)
  const { login }             = useAuth()
  const navigate              = useNavigate()

  const handleSubmit = async e => {
    e.preventDefault()
    setLoading(true)
    try {
      await login(form.email, form.password)
      toast.success('Welcome back! 🍎')
      navigate('/dashboard')
    } catch (err) {
      toast.error(err.response?.data?.message || 'Login failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  const set = k => e => setForm(p => ({ ...p, [k]: e.target.value }))

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
         style={{ background: 'radial-gradient(ellipse at 80% 50%, rgba(16,185,129,0.12) 0%, transparent 60%), radial-gradient(ellipse at 20% 80%, rgba(124,58,237,0.1) 0%, transparent 50%), #0a0f1e' }}>

      <div className="fixed top-1/4 right-1/4 w-80 h-80 rounded-full bg-primary-500/5 blur-3xl pointer-events-none" />
      <div className="fixed bottom-1/4 left-1/4 w-60 h-60 rounded-full bg-violet-500/5 blur-3xl pointer-events-none" />

      <div className="w-full max-w-md animate-slide-up">
        {/* Hero section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-cyan-400 mb-4 shadow-2xl shadow-primary-500/30 fruit-bounce">
            <Leaf className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl font-bold text-white">Welcome Back</h1>
          <p className="text-slate-400 mt-2 text-sm">Sign in to your ThanhDinh account</p>
        </div>

        {/* Demo credentials hint removed */}

        <form onSubmit={handleSubmit} className="glass-card p-8 space-y-5">
          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="login-email"
                type="email"
                value={form.email}
                onChange={set('email')}
                placeholder="your@email.com"
                required
                className="input-field pl-10"
                autoComplete="email"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="login-password"
                type={show ? 'text' : 'password'}
                value={form.password}
                onChange={set('password')}
                placeholder="Your password"
                required
                className="input-field pl-10 pr-10"
                autoComplete="current-password"
              />
              <button type="button" onClick={() => setShow(s => !s)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white">
                {show ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button id="login-submit" type="submit" disabled={loading} className="btn-primary w-full justify-center py-3 text-base">
            {loading ? <span className="spinner w-5 h-5" /> : <><LogIn size={18} /> Sign In</>}
          </button>

          <p className="text-center text-sm text-slate-500">
            Don't have an account?{' '}
            <Link to="/register" className="text-primary-400 hover:text-primary-300 font-medium">Create one</Link>
          </p>
        </form>

        {/* Features preview */}
        <div className="mt-6 grid grid-cols-3 gap-3 text-center">
          {['🍎 AI Recognition', '📊 Analytics', '📋 History'].map(f => (
            <div key={f} className="glass-card px-3 py-2.5 text-xs text-slate-400">{f}</div>
          ))}
        </div>
      </div>
    </div>
  )
}
