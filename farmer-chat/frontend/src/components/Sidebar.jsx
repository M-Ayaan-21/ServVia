import { motion, AnimatePresence } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faHeartbeat, faShieldAlt, faNotesMedical, faBolt,
  faLock, faCheckCircle, faBan, faExclamationTriangle,
} from '@fortawesome/free-solid-svg-icons'

const QUICK_QUESTIONS = [
  'Natural remedies for IBS?',
  'Safe herbs for blood pressure?',
  'Help with insomnia naturally',
  'Drug-herb interactions?',
  'Headache relief remedies',
  'Boost immunity naturally',
]

function parseArray(val) {
  if (Array.isArray(val)) return val
  if (typeof val === 'string') return val.split(',').map(s => s.trim()).filter(Boolean)
  return []
}

const SAFETY_LAYERS = [
  { icon: faExclamationTriangle, label: 'Emergency detection' },
  { icon: faBan,                 label: 'Drug-herb checker'   },
  { icon: faCheckCircle,         label: 'Critic agent review' },
]

export default function Sidebar({ profile, onQuickQuestion, onEditProfile }) {
  const allergies  = profile ? parseArray(profile.allergies) : []
  const conditions = profile ? parseArray(profile.medical_conditions) : []
  const meds       = profile ? parseArray(profile.current_medications) : []

  return (
    <motion.aside
      initial={{ x: -16, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 320, damping: 30 }}
      className="flex flex-col h-full flex-shrink-0"
      style={{
        width: '252px',
        background: 'rgba(255,255,255,0.012)',
        borderRight: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {/* ── Brand ─────────────────────────────────────────────── */}
      <div className="px-5 py-5 flex items-center gap-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{
            background: 'linear-gradient(135deg,#dc2626,#991b1b)',
            boxShadow: '0 0 16px rgba(220,38,38,0.28)',
          }}
        >
          <FontAwesomeIcon icon={faHeartbeat} className="text-white text-sm" />
        </div>
        <div>
          <p className="font-extrabold text-white text-base" style={{ letterSpacing: '-0.3px', lineHeight: 1.1 }}>ServVia</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.28)', letterSpacing: '0.08em' }}>AI HEALTHCARE</p>
        </div>
      </div>

      {/* ── Scrollable body ───────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-red px-4 py-4 space-y-5">

        {/* Profile section */}
        <AnimatePresence>
          {profile && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <SectionLabel>Profile</SectionLabel>
              <div
                className="rounded-xl p-3.5"
                style={{ background: 'rgba(220,38,38,0.05)', border: '1px solid rgba(220,38,38,0.1)' }}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white"
                      style={{ background: 'linear-gradient(135deg,#dc2626,#991b1b)' }}
                    >
                      {(profile.first_name || 'U')[0].toUpperCase()}
                    </div>
                    <span className="font-semibold text-white text-sm">{profile.first_name}</span>
                  </div>
                  <button
                    onClick={onEditProfile}
                    className="text-xs cursor-pointer px-2 py-1 rounded-lg transition-colors"
                    style={{
                      color: '#dc2626',
                      background: 'rgba(220,38,38,0.08)',
                      border: '1px solid rgba(220,38,38,0.15)',
                      fontFamily: 'inherit',
                    }}
                  >
                    Edit
                  </button>
                </div>

                <div className="space-y-1.5">
                  {allergies.length > 0 && (
                    <ProfileRow icon={faShieldAlt} color="#22c55e">
                      {allergies.length} allergen{allergies.length > 1 ? 's' : ''} protected
                    </ProfileRow>
                  )}
                  {conditions.length > 0 && (
                    <ProfileRow icon={faNotesMedical} color="#60a5fa">
                      {conditions.length} condition{conditions.length > 1 ? 's' : ''} tracked
                    </ProfileRow>
                  )}
                  {meds.length > 0 && (
                    <ProfileRow icon={faBan} color="#fbbf24">
                      {meds.length} medication{meds.length > 1 ? 's' : ''} on file
                    </ProfileRow>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Safety status */}
        <div>
          <SectionLabel>Safety Engine</SectionLabel>
          <div
            className="rounded-xl p-3.5"
            style={{ background: 'rgba(34,197,94,0.04)', border: '1px solid rgba(34,197,94,0.1)' }}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <motion.div
                className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="text-xs font-semibold" style={{ color: 'rgba(255,255,255,0.7)' }}>
                All systems active
              </span>
            </div>
            {SAFETY_LAYERS.map((l, i) => (
              <div key={i} className="flex items-center gap-2 mb-1 last:mb-0">
                <FontAwesomeIcon
                  icon={faCheckCircle}
                  style={{ color: '#22c55e', fontSize: '9px', flexShrink: 0 }}
                />
                <span className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{l.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick questions */}
        {profile && onQuickQuestion && (
          <div>
            <SectionLabel>Quick questions</SectionLabel>
            <div className="space-y-1">
              {QUICK_QUESTIONS.map((q, i) => (
                <motion.button
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * i }}
                  whileHover={{ x: 4, background: 'rgba(220,38,38,0.07)' }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => onQuickQuestion(q)}
                  className="w-full text-left text-xs px-3 py-2 rounded-lg cursor-pointer flex items-start gap-2"
                  style={{
                    background: 'rgba(255,255,255,0.025)',
                    border: '1px solid rgba(255,255,255,0.04)',
                    color: 'rgba(255,255,255,0.45)',
                    fontFamily: 'inherit',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <FontAwesomeIcon icon={faBolt} style={{ color: '#dc2626', fontSize: '8px', marginTop: '3px', flexShrink: 0 }} />
                  {q}
                </motion.button>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* ── Footer ────────────────────────────────────────────── */}
      <div className="px-4 py-3" style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
        <div className="flex items-center gap-1.5">
          <FontAwesomeIcon icon={faLock} style={{ color: 'rgba(255,255,255,0.15)', fontSize: '9px' }} />
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.15)', lineHeight: 1.4 }}>
            Not a substitute for medical advice · v3.0
          </p>
        </div>
      </div>
    </motion.aside>
  )
}

function SectionLabel({ children }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'rgba(255,255,255,0.18)', letterSpacing: '0.1em' }}>
      {children}
    </p>
  )
}

function ProfileRow({ icon, color, children }) {
  return (
    <div className="flex items-center gap-2">
      <FontAwesomeIcon icon={icon} style={{ color, fontSize: '10px', flexShrink: 0, width: '12px' }} />
      <span className="text-xs" style={{ color: 'rgba(255,255,255,0.45)' }}>{children}</span>
    </div>
  )
}
