const BASE = ''

export async function analyzeLabReport(email, files) {
  const fd = new FormData()
  fd.append('email_id', email)
  for (const f of files) fd.append('report', f)
  const res = await fetch(`${BASE}/api/lab-report/analyze/`, { method: 'POST', body: fd })
  return res.json()
}
