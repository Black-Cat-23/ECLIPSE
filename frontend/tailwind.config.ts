import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep space dark palette
        space: {
          950: '#020408',
          900: '#050A1C',
          800: '#0A1628',
          700: '#0D1F3C',
          600: '#122347',
        },
        // Nebula accent colors
        nebula: {
          blue:    '#4FC3F7',
          violet:  '#CE93D8',
          cyan:    '#80DEEA',
          indigo:  '#7986CB',
          aurora:  '#A5D6A7',
        },
        // Classification colors
        transit: '#4CAF50',    // Green — confirmed planet
        eb:      '#FF9800',    // Orange — eclipsing binary
        blend:   '#F44336',    // Red — background blend
        other:   '#9E9E9E',    // Grey — other
      },
      fontFamily: {
        sans:  ['Inter', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Outfit', 'Inter', 'sans-serif'],
      },
      animation: {
        'pulse-slow':  'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'orbit':       'orbit 20s linear infinite',
        'shimmer':     'shimmer 2s linear infinite',
        'float':       'float 6s ease-in-out infinite',
        'glow':        'glow 2s ease-in-out infinite alternate',
        'scan':        'scan 3s ease-in-out infinite',
      },
      keyframes: {
        orbit: {
          '0%':   { transform: 'rotate(0deg) translateX(120px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(120px) rotate(-360deg)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-10px)' },
        },
        glow: {
          from: { boxShadow: '0 0 10px rgba(79, 195, 247, 0.3)' },
          to:   { boxShadow: '0 0 25px rgba(79, 195, 247, 0.8), 0 0 50px rgba(79, 195, 247, 0.3)' },
        },
        scan: {
          '0%, 100%': { opacity: '0.3' },
          '50%':      { opacity: '1.0' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

export default config
