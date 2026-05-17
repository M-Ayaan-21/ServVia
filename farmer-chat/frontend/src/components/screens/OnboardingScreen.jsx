import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faClipboardList, faExclamationTriangle, faNotesMedical, faPills, faCheckCircle
} from '@fortawesome/free-solid-svg-icons'

function parseArray(val) {
  if (Array.isArray(val)) return val.join(', ')
  if (typeof val === 'string') return val
  return ''
}

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
}
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 280, damping: 24 } },
}

function Field({ label, icon, id, type = 'input', placeholder, hint, value, onChange }) {
  const inputStyle = {
    width: '100%',
    padding: '0.9rem 1.25rem',
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(220,38,38,0.15)',
    borderRadius: '14px',
    color: '#fff',
    fontSize: '1rem',
    outline: 'none',
    fontFamily: 'inherit',
    resize: type === 'textarea' ? 'vertical' : undefined,
    minHeight: type === 'textarea' ? '80px' : undefined,
    transition: 'all 0.3s',
  }

  const focusStyle = {
    borderColor: '#dc2626',
    boxShadow: '0 0 0 3px rgba(220,38,38,0.1)',
    background: 'rgba(255,255,255,0.06)',
  }

  const [focused, setFocused] = useState(false)

  return (
    <motion.div variants={itemVariants} className="mb-6 text-left">
      <label className="block mb-2 text-sm font-medium" style={{ color: 'rgba(255,255,255,0.8)' }}>
        {icon && <FontAwesomeIcon icon={icon} className="mr-2" style={{ color: '#dc2626' }} />}
        {label}
      </label>
      {type === 'textarea' ? (
        <textarea
          id={id}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ ...inputStyle, ...(focused ? focusStyle : {}) }}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
        />
      ) : (
        <input
          type={type}
          id={id}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ ...inputStyle, ...(focused ? focusStyle : {}) }}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
        />
      )}
      {hint && <p className="mt-1 text-xs" style={{ color: 'var(--text-muted)' }}>{hint}</p>}
    </motion.div>
  )
}

export default function OnboardingScreen({ isEdit, existingProfile, onSave, loading }) {
  const [name, setName] = useState('')
  const [allergies, setAllergies] = useState('')
  const [conditions, setConditions] = useState('')
  const [medications, setMedications] = useState('')

  useEffect(() => {
    if (existingProfile) {
      setName(existingProfile.first_name || '')
      setAllergies(parseArray(existingProfile.allergies))
      setConditions(parseArray(existingProfile.medical_conditions))
      setMedications(parseArray(existingProfile.current_medications))
    }
  }, [existingProfile])

  function submit() {
    if (!name.trim()) return
    onSave({ name, allergies, conditions, medications })
  }

  return (
    <motion.div
      className="flex flex-col items-center justify-center text-center min-h-full py-12"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      <motion.div
        variants={itemVariants}
        animate={{ y: [0, -12, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className="text-5xl mb-6"
      >
        <FontAwesomeIcon
          icon={faClipboardList}
          style={{
            background: 'linear-gradient(135deg,#fff,#dc2626)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        />
      </motion.div>

      <motion.h2 variants={itemVariants} className="text-4xl font-extrabold text-gradient mb-3" style={{ letterSpacing: '-1px' }}>
        {isEdit ? 'Update Your Profile' : 'Tell Us About Yourself'}
      </motion.h2>
      <motion.p variants={itemVariants} className="text-base mb-8" style={{ color: 'var(--text-muted)' }}>
        This helps us provide safe, personalized health recommendations
      </motion.p>

      <div className="w-full max-w-md">
        <Field label="What's your name?" id="nameInput" placeholder="Your first name" value={name} onChange={setName} />
        <Field
          label="Do you have any allergies?"
          icon={faExclamationTriangle}
          id="allergiesInput"
          type="textarea"
          placeholder="e.g., peanuts, shellfish, penicillin (or leave blank)"
          hint="Separate multiple allergies with commas"
          value={allergies}
          onChange={setAllergies}
        />
        <Field
          label="Any existing medical conditions?"
          icon={faNotesMedical}
          id="conditionsInput"
          type="textarea"
          placeholder="e.g., diabetes, hypertension (or leave blank)"
          hint="This helps us recommend safe remedies"
          value={conditions}
          onChange={setConditions}
        />
        <Field
          label="Current medications?"
          icon={faPills}
          id="medicationsInput"
          type="textarea"
          placeholder="e.g., aspirin, metformin (or leave blank)"
          hint="We'll check for potential interactions"
          value={medications}
          onChange={setMedications}
        />

        <motion.button
          variants={itemVariants}
          whileHover={{ translateY: -2, boxShadow: '0 8px 30px rgba(220,38,38,0.45)' }}
          whileTap={{ scale: 0.98 }}
          onClick={submit}
          disabled={loading || !name.trim()}
          className="w-full py-4 rounded-2xl text-white font-semibold text-base cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          style={{
            background: 'linear-gradient(135deg,#dc2626,#991b1b)',
            border: 'none',
            boxShadow: '0 4px 20px rgba(220,38,38,0.35)',
            fontFamily: 'inherit',
          }}
        >
          <FontAwesomeIcon icon={faCheckCircle} />
          {loading ? 'Saving...' : 'Save Profile & Continue'}
        </motion.button>
      </div>
    </motion.div>
  )
}
