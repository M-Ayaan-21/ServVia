#!/bin/bash
cd "$(dirname "$0")"

# ── Django backend ────────────────────────────────────────────────────────
echo "Starting ServVia backend on port 8001..."
source .myenv/bin/activate
export PYTHONWARNINGS="ignore"
python manage.py runserver 8001 &
DJANGO_PID=$!

# ── Vite frontend ─────────────────────────────────────────────────────────
echo "Starting ServVia frontend on port 5173..."
cd frontend
npm run dev &
VITE_PID=$!
cd ..

echo ""
echo "ServVia is running:"
echo "  Frontend  →  http://localhost:5173"
echo "  Backend   →  http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop both servers."

# ── Graceful shutdown on Ctrl+C ───────────────────────────────────────────
trap "echo ''; echo 'Stopping...'; kill $DJANGO_PID $VITE_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
