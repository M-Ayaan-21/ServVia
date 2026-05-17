const BASE = ''

export async function analyzeSkin(email, imageFile) {
  const fd = new FormData()
  fd.append('email_id', email)
  fd.append('image', imageFile)
  const res = await fetch(`${BASE}/api/skin/analyze/`, { method: 'POST', body: fd })
  return res.json()
}
