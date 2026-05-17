import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faClock, faLocationDot, faCloudSun } from '@fortawesome/free-solid-svg-icons'

function fmt(str) {
  return (str || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export default function ChronoCard({ bioState }) {
  if (!bioState) return null

  const tags = [
    fmt(bioState.circadian_phase),
    fmt(bioState.seasonal_influence),
    `Sleep Pressure: ${fmt(bioState.sleep_pressure)}`,
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="mt-3 p-3 rounded-xl"
      style={{ background: 'rgba(99,102,241,0.04)', border: '1px solid rgba(99,102,241,0.15)' }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <FontAwesomeIcon icon={faClock} style={{ color: '#818cf8', fontSize: '0.8rem' }} />
          <span className="text-xs font-semibold" style={{ color: 'rgba(255,255,255,0.75)' }}>
            Biological Context
          </span>
        </div>

        {/* Location + weather pill */}
        {(bioState.location || bioState.weather) && (
          <div className="flex items-center gap-2">
            {bioState.location && (
              <span className="flex items-center gap-1 text-xs" style={{ color: 'rgba(165,168,252,0.7)', fontSize: '0.65rem' }}>
                <FontAwesomeIcon icon={faLocationDot} style={{ fontSize: '0.6rem' }} />
                {bioState.location}
              </span>
            )}
            {bioState.weather && (
              <span className="flex items-center gap-1 text-xs" style={{ color: 'rgba(165,168,252,0.7)', fontSize: '0.65rem' }}>
                <FontAwesomeIcon icon={faCloudSun} style={{ fontSize: '0.6rem' }} />
                {bioState.weather}
                {bioState.temperature_celsius != null && ` · ${bioState.temperature_celsius.toFixed(0)}°C`}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Circadian tags */}
      <div>
        {tags.map((tag, i) => (
          <span key={i} className="inline-block px-2 py-0.5 rounded mr-1 mt-1"
            style={{ background: 'rgba(99,102,241,0.08)', color: 'rgba(165,168,252,0.9)', fontSize: '0.68rem' }}>
            {tag}
          </span>
        ))}
        {bioState.is_misaligned && (
          <span className="inline-block px-2 py-0.5 rounded mr-1 mt-1"
            style={{ background: 'rgba(239,68,68,0.1)', color: '#fca5a5', fontSize: '0.68rem' }}>
            ⚠ Circadian Misalignment
          </span>
        )}
      </div>
    </motion.div>
  )
}
