import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faExclamationTriangle } from '@fortawesome/free-solid-svg-icons'

export default function SafetyCard({ safety }) {
  if (!safety || safety.is_safe) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.35 }}
      className="mt-3 p-3 rounded-xl"
      style={{ background: 'rgba(239,68,68,0.04)', border: '1px solid rgba(239,68,68,0.2)' }}
    >
      <div className="font-semibold text-xs mb-1" style={{ color: '#fbbf24' }}>
        <FontAwesomeIcon icon={faExclamationTriangle} className="mr-1" />
        Safety Details
      </div>
      <div className="text-xs" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        {safety.blocked_herb && <div><strong style={{ color: '#fff' }}>Blocked:</strong> {safety.blocked_herb}</div>}
        {safety.washout_days_remaining && <div><strong style={{ color: '#fff' }}>Washout:</strong> {safety.washout_days_remaining} day(s)</div>}
        {safety.contraindications?.length > 0 && (
          <div><strong style={{ color: '#fff' }}>Contraindications:</strong> {safety.contraindications.join(', ')}</div>
        )}
      </div>
    </motion.div>
  )
}
