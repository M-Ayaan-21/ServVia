const BASE = ''

export async function checkProfile(email) {
  const res = await fetch(`${BASE}/api/profile/profile/check_profile/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email_id: email }),
  })
  return res.json()
}

export async function saveProfile({ email, name, allergies, conditions, medications }) {
  const res = await fetch(`${BASE}/api/profile/profile/create_or_update_profile/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email_id: email,
      first_name: name,
      allergies,
      medical_conditions: conditions,
      current_medications: medications,
    }),
  })
  return res.json()
}
