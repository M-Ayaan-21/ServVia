import { useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCamera, faImages, faFileUpload } from '@fortawesome/free-solid-svg-icons'

export default function OptionsMenu({ open, onClose, onSkinImage, onLabFiles }) {
  const skinGalleryRef = useRef()
  const skinCameraRef = useRef()
  const labFilesRef = useRef()
  const labCameraRef = useRef()

  function handleSkinGallery(e) {
    const f = e.target.files[0]; if (!f) return
    if (f.size > 5 * 1024 * 1024) { alert('Max 5MB'); return }
    onSkinImage(f); onClose(); e.target.value = ''
  }
  function handleLabFiles(e) {
    const fs = Array.from(e.target.files); if (!fs.length) return
    for (const f of fs) if (f.size > 10 * 1024 * 1024) { alert(`${f.name} too large. Max 10MB`); return }
    onLabFiles(fs); onClose(); e.target.value = ''
  }

  const sectionLabel = (text) => (
    <div className="px-4 py-2 text-xs uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
      {text}
    </div>
  )

  const menuItem = (icon, label, onClick) => (
    <motion.button
      whileHover={{ background: 'rgba(220,38,38,0.08)' }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="flex items-center gap-3 px-4 py-3 w-full text-left rounded-xl text-sm cursor-pointer transition-colors"
      style={{ color: 'rgba(255,255,255,0.8)', background: 'transparent', border: 'none', fontFamily: 'inherit' }}
    >
      <FontAwesomeIcon icon={icon} style={{ color: '#dc2626', width: '22px' }} />
      {label}
    </motion.button>
  )

  return (
    <>
      {/* Hidden file inputs */}
      <input ref={skinGalleryRef} type="file" accept="image/*" className="hidden" onChange={handleSkinGallery} />
      <input ref={skinCameraRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleSkinGallery} />
      <input ref={labFilesRef} type="file" multiple className="hidden" onChange={handleLabFiles} />
      <input ref={labCameraRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleLabFiles} />

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 10 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
            className="absolute bottom-16 left-0 rounded-2xl py-2 min-w-52"
            style={{
              background: 'rgba(10,10,10,0.97)',
              border: '1px solid rgba(220,38,38,0.25)',
              boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
              transformOrigin: 'bottom left',
              zIndex: 20,
            }}
          >
            {sectionLabel('Skin Analysis')}
            {menuItem(faCamera, 'Take Photo', () => skinCameraRef.current?.click())}
            {menuItem(faImages, 'Choose from Gallery', () => skinGalleryRef.current?.click())}

            <div className="mx-4 my-1 h-px" style={{ background: 'rgba(220,38,38,0.15)' }} />

            {sectionLabel('Lab Report')}
            {menuItem(faCamera, 'Take Photo', () => labCameraRef.current?.click())}
            {menuItem(faFileUpload, 'Upload Files', () => labFilesRef.current?.click())}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
