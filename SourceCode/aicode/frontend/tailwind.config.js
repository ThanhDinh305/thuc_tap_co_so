/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        dark: {
          900: '#0a0f1e',
          800: '#0f172a',
          700: '#1e293b',
          600: '#334155',
          500: '#475569',
        },
        accent: {
          purple: '#7c3aed',
          violet: '#8b5cf6',
          cyan:   '#06b6d4',
          amber:  '#f59e0b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Plus Jakarta Sans', 'sans-serif'],
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'fade-in':     'fadeIn 0.3s ease-in-out',
        'slide-up':    'slideUp 0.4s ease-out',
        'slide-in':    'slideIn 0.3s ease-out',
        'pulse-slow':  'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow':   'spin 3s linear infinite',
        'bounce-slow': 'bounce 2s infinite',
        'glow':        'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn:  { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp: { '0%': { transform: 'translateY(20px)', opacity: '0' }, '100%': { transform: 'translateY(0)', opacity: '1' } },
        slideIn: { '0%': { transform: 'translateX(-20px)', opacity: '0' }, '100%': { transform: 'translateX(0)', opacity: '1' } },
        glow:    { '0%': { boxShadow: '0 0 5px rgba(16,185,129,0.2)' }, '100%': { boxShadow: '0 0 20px rgba(16,185,129,0.6)' } },
      },
    },
  },
  plugins: [],
}
