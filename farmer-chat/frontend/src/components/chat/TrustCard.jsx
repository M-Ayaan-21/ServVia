import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicroscope, faCheckCircle, faQuestionCircle, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons'

export default function TrustCard({ trust }) {
  if (!trust) return null
  const { verified_count: v = 0, unverified_count: u = 0, interaction_warnings: w = [] } = trust
  if (!v && !u && !w.length) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25 }}
      className="mt-3 p-3 rounded-xl"
      style={{ background: 'rgba(20,184,166,0.04)', border: '1px solid rgba(20,184,166,0.15)' }}
    >
      <div className="flex items-center gap-2 mb-2">
        <FontAwesomeIcon icon={faMicroscope} style={{ color: '#14b8a6', fontSize: '0.85rem' }} />
        <span className="text-xs font-semibold" style={{ color: 'rgba(255,255,255,0.8)' }}>Scientific Validation</span>
      </div>
      <div className="flex gap-2 flex-wrap">
        {v > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-xl text-xs font-semibold"
            style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.2)', color: '#22c55e' }}>
            <FontAwesomeIcon icon={faCheckCircle} /> {v} Verified
          </span>
        )}
        {u > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-xl text-xs font-semibold"
            style={{ background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.2)', color: '#fbbf24' }}>
            <FontAwesomeIcon icon={faQuestionCircle} /> {u} Unverified
          </span>
        )}
        {w.length > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-xl text-xs font-semibold"
            style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444' }}>
            <FontAwesomeIcon icon={faExclamationTriangle} /> {w.length} Alert{w.length > 1 ? 's' : ''}
          </span>
        )}
      </div>
    </motion.div>
  )
}
