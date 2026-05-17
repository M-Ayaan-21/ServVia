import { motion } from 'framer-motion'

export default function StatusDot() {
  return (
    <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
      <motion.div
        className="w-2 h-2 rounded-full bg-green-500"
        animate={{ opacity: [1, 0.7, 1], boxShadow: ['0 0 0 0 rgba(34,197,94,0.4)', '0 0 0 6px rgba(34,197,94,0)', '0 0 0 0 rgba(34,197,94,0.4)'] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
      />
      <span>Online</span>
    </div>
  )
}
