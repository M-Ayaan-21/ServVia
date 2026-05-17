import { useState } from 'react'
import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faStethoscope, faRobot, faShieldAlt, faLeaf } from '@fortawesome/free-solid-svg-icons'

const features = [
  { icon: faRobot, label: 'Multi-Agent AI' },
  { icon: faShieldAlt, label: 'Safety Verified' },
  { icon: faLeaf, label: 'Natural Remedies' },
]

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
}
const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } },
}

export default function EmailScreen({ onContinue, loading }) {
  const [email, setEmail] = useState('')

  function submit() {
    if (!email || !email.includes('@')) return
    onContinue(email)
  }

  return (
    <motion.div
      className="flex flex-col items-center justify-center text-center min-h-full py-12"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* Icon */}
      <motion.div
        variants={itemVariants}
        animate={{ y: [0, -12, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className="text-6xl mb-8"
      >
        <FontAwesomeIcon
          icon={faStethoscope}
          style={{
            background: 'linear-gradient(135deg,#fff,#dc2626)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        />
      </motion.div>

      {/* Title */}
      <motion.h2 variants={itemVariants} className="text-4xl font-extrabold text-gradient mb-3" style={{ letterSpacing: '-1px' }}>
        Welcome to ServVia
      </motion.h2>
      <motion.p variants={itemVariants} className="text-lg mb-10" style={{ color: 'var(--text-muted)' }}>
        Your Personalized AI Health Companion
      </motion.p>

      {/* Form */}
      <motion.div variants={itemVariants} className="w-full max-w-md">
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          placeholder="Enter your email address"
          className="w-full px-5 py-4 rounded-2xl text-base text-white outline-none mb-4 transition-all duration-300"
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(220,38,38,0.15)',
            fontFamily: 'inherit',
          }}
          onFocus={e => {
            e.target.style.borderColor = '#dc2626'
            e.target.style.boxShadow = '0 0 0 3px rgba(220,38,38,0.1)'
          }}
          onBlur={e => {
            e.target.style.borderColor = 'rgba(220,38,38,0.15)'
            e.target.style.boxShadow = 'none'
          }}
        />
        <motion.button
          whileHover={{ translateY: -2, boxShadow: '0 8px 30px rgba(220,38,38,0.45)' }}
          whileTap={{ scale: 0.98 }}
          onClick={submit}
          disabled={loading}
          className="w-full py-4 rounded-2xl text-white font-semibold text-base cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          style={{
            background: 'linear-gradient(135deg,#dc2626,#991b1b)',
            border: 'none',
            boxShadow: '0 4px 20px rgba(220,38,38,0.35)',
            fontFamily: 'inherit',
          }}
        >
          {loading ? 'Checking...' : 'Continue'}
        </motion.button>
      </motion.div>

      {/* Feature cards */}
      <motion.div variants={containerVariants} className="grid grid-cols-3 gap-5 mt-12 max-w-lg w-full">
        {features.map((f, i) => (
          <motion.div
            key={i}
            variants={itemVariants}
            whileHover={{
              translateY: -6,
              borderColor: 'rgba(220,38,38,0.35)',
              boxShadow: '0 12px 40px rgba(220,38,38,0.1)',
            }}
            className="flex flex-col items-center p-6 rounded-2xl cursor-default transition-colors"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(220,38,38,0.15)',
            }}
          >
            <FontAwesomeIcon icon={f.icon} className="text-3xl mb-3" style={{ color: '#dc2626' }} />
            <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>{f.label}</p>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  )
}
