import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { toast } from 'react-hot-toast'
import { Leaf, Eye, EyeOff, Mail, Lock, User, ArrowRight, Sparkles } from 'lucide-react'

export default function Register() {
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' })
  const [show, setShow] = useState(false)
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async e => {
    e.preventDefault()
    if (form.password !== form.confirm)
      return toast.error('Passwords do not match.')
    setLoading(true)
    try {
      await register(form.name, form.email, form.password)
      toast.success('Account created! Welcome to ThanhDinh 🎉')
      navigate('/dashboard')
    } catch (err) {
      toast.error(err.response?.data?.message || 'Registration failed.')
    } finally {
      setLoading(false)
    }
  }

  const set = k => e => setForm(p => ({ ...p, [k]: e.target.value }))

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
         style={{ background: 'radial-gradient(ellipse at 20% 50%, rgba(16,185,129,0.15) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(124,58,237,0.1) 0%, transparent 50%), #0a0f1e' }}>

      {/* Decorative orbs */}
      <div className="fixed top-20 left-20 w-72 h-72 rounded-full bg-primary-500/5 blur-3xl pointer-events-none" />
      <div className="fixed bottom-20 right-20 w-96 h-96 rounded-full bg-violet-500/5 blur-3xl pointer-events-none" />

      <div className="w-full max-w-md animate-slide-up">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-cyan-400 mb-4 shadow-2xl shadow-primary-500/30">
            <Leaf className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl font-bold text-white">Create Account</h1>
          <p className="text-slate-400 mt-2 text-sm">Join ThanhDinh and start recognizing fruits</p>
        </div>

        <form onSubmit={handleSubmit} className="glass-card p-8 space-y-5">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Full Name</label>
            <div className="relative">
              <User size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="reg-name"
                type="text"
                value={form.name}
                onChange={set('name')}
                placeholder="John Doe"
                required
                className="input-field pl-10"
              />
            </div>
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="reg-email"
                type="email"
                value={form.email}
                onChange={set('email')}
                placeholder="john@example.com"
                required
                className="input-field pl-10"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="reg-password"
                type={show ? 'text' : 'password'}
                value={form.password}
                onChange={set('password')}
                placeholder="Min. 6 characters"
                required
                minLength={6}
                className="input-field pl-10 pr-10"
              />
              <button type="button" onClick={() => setShow(s => !s)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white">
                {show ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Confirm */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Confirm Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                id="reg-confirm"
                type={show ? 'text' : 'password'}
                value={form.confirm}
                onChange={set('confirm')}
                placeholder="Repeat password"
                required
                className="input-field pl-10"
              />
            </div>
          </div>

          <button id="reg-submit" type="submit" disabled={loading} className="btn-primary w-full justify-center py-3 text-base">
            {loading ? <span className="spinner w-5 h-5" /> : <><Sparkles size={18} /> Create Account</>}
          </button>

          <p className="text-center text-sm text-slate-500">
            Already have an account?{' '}
            <Link to="/login" className="text-primary-400 hover:text-primary-300 font-medium">Sign in</Link>
          </p>
        </form>
      </div>
    </div>
  )
}
