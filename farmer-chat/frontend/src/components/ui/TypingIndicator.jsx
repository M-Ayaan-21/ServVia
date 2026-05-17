import { motion } from 'framer-motion'

export default function TypingIndicator({ label = 'ServVia is analyzing...' }) {
  return (
    <div className="flex items-center gap-3 px-6 py-4">
      <div className="flex gap-1">
        {[0, 1, 2].map(i => (
          <motion.span
            key={i}
            className="w-2 h-2 rounded-full bg-red"
            animate={{ scale: [0.6, 1, 0.6], opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.15, ease: 'easeInOut' }}
          />
        ))}
      </div>
      <span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}
