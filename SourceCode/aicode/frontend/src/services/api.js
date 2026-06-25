import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// Attach JWT token to every request
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Handle 401 globally → logout
api.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ── Auth ────────────────────────────────────────────────────────────────────
export const authApi = {
  register:      data => api.post('/auth/register', data),
  login:         data => api.post('/auth/login', data),
  getMe:         ()   => api.get('/auth/me'),
  updateProfile: data => api.put('/auth/profile', data),
}

// ── Predictions ─────────────────────────────────────────────────────────────
export const predictApi = {
  upload:  formData => api.post('/predictions/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 60000 }),
  webcam:  data     => api.post('/predictions/webcam',  data, { timeout: 60000 }),
}

// ── History ─────────────────────────────────────────────────────────────────
export const historyApi = {
  getAll:      params => api.get('/history', { params }),
  getOne:      id     => api.get(`/history/${id}`),
  deleteOne:   id     => api.delete(`/history/${id}`),
  deleteAll:   ()     => api.delete('/history'),
  exportExcel: ()     => api.get('/history/export/excel', { responseType: 'blob' }),
  exportPDF:   ()     => api.get('/history/export/pdf',   { responseType: 'blob' }),
}

// ── Dashboard ────────────────────────────────────────────────────────────────
export const dashboardApi = {
  getStats:        () => api.get('/dashboard/stats'),
  getWeekly:       () => api.get('/dashboard/weekly'),
  getMonthly:      () => api.get('/dashboard/monthly'),
  getDistribution: () => api.get('/dashboard/distribution'),
}

// ── Admin ────────────────────────────────────────────────────────────────────
export const adminApi = {
  getUsers:      params => api.get('/admin/users',   { params }),
  deleteUser:    id     => api.delete(`/admin/users/${id}`),
  getRecords:    params => api.get('/admin/records', { params }),
  deleteRecord:  id     => api.delete(`/admin/records/${id}`),
  getStats:      ()     => api.get('/admin/stats'),
}

export default api
