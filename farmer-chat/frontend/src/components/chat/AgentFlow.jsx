import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPen, faSearch, faCheck, faTimesCircle } from '@fortawesome/free-solid-svg-icons'

export default function AgentFlow({ blocked }) {
  const nodes = blocked
    ? [
        { label: 'Proposer', icon: faPen, cls: 'proposer' },
        { label: 'Critic', icon: faSearch, cls: 'critic' },
        { label: 'Blocked by Critic', icon: faTimesCircle, emergency: true },
      ]
    : [
        { label: 'Proposer', icon: faPen, cls: 'proposer' },
        { label: 'Critic', icon: faSearch, cls: 'critic' },
        { label: 'Approved', icon: faCheck, cls: 'approved' },
      ]

  const nodeStyles = {
    proposer: { background: 'rgba(59,130,246,0.12)', color: '#60a5fa' },
    critic: { background: 'rgba(168,85,247,0.12)', color: '#c084fc' },
    approved: { background: 'rgba(34,197,94,0.12)', color: '#4ade80' },
    emergency: { background: 'rgba(239,68,68,0.12)', color: '#ef4444' },
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="flex items-center gap-2 mt-3 px-3 py-2 rounded-xl text-xs"
      style={{
        background: 'rgba(139,92,246,0.04)',
        border: '1px solid rgba(139,92,246,0.15)',
        color: 'rgba(255,255,255,0.5)',
      }}
    >
      {nodes.map((n, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <span style={{ color: 'rgba(255,255,255,0.2)' }}>→</span>}
          <span
            className="px-2 py-1 rounded-md font-semibold flex items-center gap-1"
            style={nodeStyles[n.cls] || nodeStyles.emergency}
          >
            <FontAwesomeIcon icon={n.icon} />
            {n.label}
          </span>
        </span>
      ))}
    </motion.div>
  )
}
