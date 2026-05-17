import { motion, AnimatePresence } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faHeartbeat, faShieldAlt, faUserEdit, faNotesMedical } from '@fortawesome/free-solid-svg-icons'
import StatusDot from './ui/StatusDot'

function parseArray(val) {
  if (Array.isArray(val)) return val
  if (typeof val === 'string') return val.split(',').map(s => s.trim()).filter(Boolean)
  return []
}

export default function Header({ profile, onEditProfile, compact = false, minimal = false }) {
  const allergies  = profile ? parseArray(profile.allergies) : []
  const conditions = profile ? parseArray(profile.medical_conditions) : []

  return (
    <header
      className="flex items-center justify-between px-5 border-b flex-shrink-0"
      style={{
        height: compact ? '52px' : '68px',
        borderColor: 'rgba(255,255,255,0.05)',
        background: compact
          ? 'rgba(5,5,5,0.7)'
          : 'rgba(5,5,5,0.85)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        position: 'relative',
        zIndex: 10,
        transition: 'height 0.3s ease',
      }}
    >
      {/* Logo — only when not compact (auth screens, or minimal) */}
      {!compact && (
        <div className="flex items-center gap-3">
          <motion.div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#dc2626,#991b1b)', boxShadow: '0 0 16px rgba(220,38,38,0.35)' }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <FontAwesomeIcon icon={faHeartbeat} className="text-white text-sm" />
          </motion.div>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-gradient" style={{ letterSpacing: '-0.4px', lineHeight: 1.1 }}>
              ServVia
            </h1>
            <p className="text-xs tracking-widest uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              AI Healthcare
            </p>
          </div>
        </div>
      )}

      {/* Compact mode: left side shows conversation context */}
      {compact && (
        <div className="flex items-center gap-2">
          <StatusDot />
          <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.35)' }}>
            Multi-Agent · Active
          </span>
        </div>
      )}

      {/* Right side */}
      <div className="flex items-center gap-4">
        <AnimatePresence>
          {profile && !compact && (
            <motion.div
              className="flex items-center gap-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            >
              {allergies.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium"
                  style={{
                    background: 'rgba(220,38,38,0.08)',
                    border: '1px solid rgba(220,38,38,0.25)',
                    color: '#dc2626',
                  }}
                >
                  <FontAwesomeIcon icon={faShieldAlt} className="text-xs" />
                  {allergies.length} allergen{allergies.length > 1 ? 's' : ''} protected
                </motion.div>
              )}
              {conditions.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.05 }}
                  className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium"
                  style={{
                    background: 'rgba(220,38,38,0.08)',
                    border: '1px solid rgba(220,38,38,0.25)',
                    color: '#dc2626',
                  }}
                >
                  <FontAwesomeIcon icon={faNotesMedical} className="text-xs" />
                  {conditions.length} condition{conditions.length > 1 ? 's' : ''}
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* In compact mode, show user name + edit button inline and small */}
        {compact && profile && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>
              {profile.first_name}
            </span>
            <motion.button
              whileHover={{ background: 'rgba(220,38,38,0.15)' }}
              whileTap={{ scale: 0.95 }}
              onClick={onEditProfile}
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium cursor-pointer"
              style={{
                background: 'rgba(220,38,38,0.07)',
                border: '1px solid rgba(220,38,38,0.18)',
                color: 'rgba(220,38,38,0.8)',
                fontFamily: 'inherit',
              }}
            >
              <FontAwesomeIcon icon={faUserEdit} style={{ fontSize: '10px' }} />
              Edit
            </motion.button>
          </div>
        )}

        {!compact && profile && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            whileHover={{ scale: 1.05, background: 'rgba(220,38,38,0.15)' }}
            whileTap={{ scale: 0.95 }}
            onClick={onEditProfile}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium cursor-pointer"
            style={{
              background: 'rgba(220,38,38,0.08)',
              border: '1px solid rgba(220,38,38,0.25)',
              color: '#dc2626',
              fontFamily: 'inherit',
            }}
          >
            <FontAwesomeIcon icon={faUserEdit} />
            Edit Profile
          </motion.button>
        )}

        {!compact && <StatusDot />}
      </div>
    </header>
  )
}
