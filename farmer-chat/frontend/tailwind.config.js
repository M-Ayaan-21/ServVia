/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        red: {
          DEFAULT: '#dc2626',
          dark: '#991b1b',
          light: '#ef4444',
        },
        bg: '#050505',
        surface: 'rgba(255,255,255,0.03)',
        'surface-hover': 'rgba(255,255,255,0.06)',
      },
      backgroundImage: {
        'red-gradient': 'linear-gradient(135deg, #dc2626, #991b1b)',
        'text-gradient': 'linear-gradient(135deg, #ffffff 30%, #dc2626 100%)',
      },
      boxShadow: {
        'red-sm': '0 0 16px rgba(220,38,38,0.35)',
        'red-md': '0 4px 20px rgba(220,38,38,0.35)',
        'red-lg': '0 8px 30px rgba(220,38,38,0.45)',
        'red-glow': '0 0 30px rgba(220,38,38,0.06)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

