import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faShieldAlt, faExclamationTriangle, faBan, faRobot } from '@fortawesome/free-solid-svg-icons'

const variants = {
  safe:               { bg: 'rgba(34,197,94,0.1)',   border: 'rgba(34,197,94,0.25)',   color: '#22c55e', icon: faShieldAlt,          label: 'Verified Safe' },
  emergency_intercept:{ bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',   color: '#ef4444', icon: faExclamationTriangle, label: 'Emergency Protocol' },
  safety_blocked:     { bg: 'rgba(251,191,36,0.1)',  border: 'rgba(251,191,36,0.25)',  color: '#fbbf24', icon: faBan,                label: 'Safety Blocked' },
  emergency_triage:   { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',   color: '#ef4444', icon: faExclamationTriangle, label: 'Triage Alert — Medical Evaluation Required' },
  agent:              { bg: 'rgba(139,92,246,0.1)',  border: 'rgba(139,92,246,0.25)', color: '#a78bfa', icon: faRobot,              label: 'Agent Verified' },
}

export default function PipelineBadge({ type }) {
  const v = variants[type] || variants.safe
  const isEmergency = type === 'emergency_intercept' || type === 'emergency_triage'

  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold mr-2"
      style={{
        background: v.bg,
        border: `1px solid ${v.border}`,
        color: v.color,
        animation: isEmergency ? 'emergencyPulse 1.5s infinite' : undefined,
      }}
    >
      <FontAwesomeIcon icon={v.icon} />
      {v.label}

      {isEmergency && (
        <style>{`
          @keyframes emergencyPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
            50%       { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
          }
        `}</style>
      )}
    </motion.span>
  )
}
