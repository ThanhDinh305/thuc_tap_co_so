import { useState } from 'react'
import { useAuth } from '../../context/AuthContext'
import { authApi } from '../../services/api'
import { toast } from 'react-hot-toast'
import { User, Mail, Lock, Save, Shield } from 'lucide-react'
import { format } from 'date-fns'

export default function Profile() {
  const { user, updateUser } = useAuth()
  const [nameForm, setNameForm]       = useState({ name: user?.name || '' })
  const [passForm, setPassForm]       = useState({ currentPassword: '', newPassword: '', confirm: '' })
  const [nameLoading, setNameLoading] = useState(false)
  const [passLoading, setPassLoading] = useState(false)
  const [showPass,    setShowPass]    = useState(false)

  const handleUpdateName = async e => {
    e.preventDefault()
    setNameLoading(true)
    try {
      const res = await authApi.updateProfile({ name: nameForm.name })
      updateUser(res.data.user)
      toast.success('Name updated!')
    } catch (err) {
      toast.error(err.response?.data?.message || 'Update failed.')
    } finally { setNameLoading(false) }
  }

  const handleUpdatePassword = async e => {
    e.preventDefault()
    if (passForm.newPassword !== passForm.confirm)
      return toast.error('New passwords do not match.')
    setPassLoading(true)
    try {
      await authApi.updateProfile({ currentPassword: passForm.currentPassword, newPassword: passForm.newPassword })
      setPassForm({ currentPassword: '', newPassword: '', confirm: '' })
      toast.success('Password updated!')
    } catch (err) {
      toast.error(err.response?.data?.message || 'Update failed.')
    } finally { setPassLoading(false) }
  }

  return (
    <div className="max-w-2xl space-y-6 animate-fade-in">
      {/* Profile header */}
      <div className="glass-card p-6 flex items-center gap-5"
           style={{ background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(124,58,237,0.08))' }}>
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-500 to-violet-500 flex items-center justify-center text-3xl font-bold text-white shadow-xl shadow-primary-500/30 flex-shrink-0">
          {user?.name?.charAt(0)?.toUpperCase() || 'U'}
        </div>
        <div className="flex-1">
          <h2 className="font-display text-2xl font-bold text-white">{user?.name}</h2>
          <p className="text-slate-400 text-sm">{user?.email}</p>
          <div className="flex items-center gap-2 mt-2">
            <span className={user?.role === 'admin' ? 'badge badge-purple' : 'badge badge-green'}>
              <Shield size={10} className="mr-1" />
              {user?.role === 'admin' ? 'Administrator' : 'User'}
            </span>
            {user?.created_at && (
              <span className="text-xs text-slate-500">
                Member since {format(new Date(user.created_at), 'MMM yyyy')}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Update name */}
      <form onSubmit={handleUpdateName} className="glass-card p-6 space-y-4">
        <h3 className="section-title flex items-center gap-2"><User size={20} className="text-primary-400" /> Personal Info</h3>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Display Name</label>
          <input
            id="profile-name"
            type="text"
            value={nameForm.name}
            onChange={e => setNameForm({ name: e.target.value })}
            required
            className="input-field"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Email Address</label>
          <div className="relative">
            <Mail size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input type="email" value={user?.email || ''} disabled className="input-field pl-10 opacity-50 cursor-not-allowed" />
          </div>
          <p className="text-xs text-slate-500 mt-1">Email cannot be changed.</p>
        </div>

        <button id="save-name" type="submit" disabled={nameLoading} className="btn-primary">
          {nameLoading ? <span className="spinner w-4 h-4" /> : <><Save size={16} /> Save Changes</>}
        </button>
      </form>

      {/* Change password */}
      <form onSubmit={handleUpdatePassword} className="glass-card p-6 space-y-4">
        <h3 className="section-title flex items-center gap-2"><Lock size={20} className="text-violet-400" /> Change Password</h3>

        {[
          { id: 'current-pass',  label: 'Current Password',  key: 'currentPassword', ph: 'Enter current password' },
          { id: 'new-pass',      label: 'New Password',      key: 'newPassword',     ph: 'Min. 6 characters'     },
          { id: 'confirm-pass',  label: 'Confirm Password',  key: 'confirm',         ph: 'Repeat new password'   },
        ].map(({ id, label, key, ph }) => (
          <div key={key}>
            <label className="block text-sm font-medium text-slate-300 mb-2">{label}</label>
            <input
              id={id}
              type={showPass ? 'text' : 'password'}
              value={passForm[key]}
              onChange={e => setPassForm(p => ({ ...p, [key]: e.target.value }))}
              placeholder={ph}
              required
              minLength={key !== 'currentPassword' ? 6 : 1}
              className="input-field"
            />
          </div>
        ))}

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="show-password"
            checked={showPass}
            onChange={e => setShowPass(e.target.checked)}
            className="rounded text-primary-500"
          />
          <label htmlFor="show-password" className="text-sm text-slate-400 cursor-pointer">Show passwords</label>
        </div>

        <button id="save-password" type="submit" disabled={passLoading} className="btn-primary">
          {passLoading ? <span className="spinner w-4 h-4" /> : <><Lock size={16} /> Update Password</>}
        </button>
      </form>
    </div>
  )
}
