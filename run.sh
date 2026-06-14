#!/bin/bash
set -e

echo "🔬 ImmunoWatch — AI Health Monitoring System"
echo "============================================="

# --- resolve a usable Python interpreter (python3, else python) ---
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 required"; exit 1
fi
command -v npm >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }

echo ""
echo "📊 Step 1/5: Generating patient biosignal data..."
$PY data/simulator.py

echo ""
echo "🧠 Step 2/5: Training personal baseline models (LSTM Autoencoder)..."
$PY ml/baseline.py

echo ""
echo "🔮 Step 3/5: Training infection risk predictor (Temporal Transformer)..."
$PY ml/predictor.py

echo ""
echo "🌐 Step 4/5: Running federated learning simulation..."
$PY ml/federated.py

echo ""
echo "📈 Generating evaluation report..."
$PY ml/evaluation.py

echo ""
echo "🚀 Step 5/5: Starting services..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "⏳ Waiting for API to initialize..."
sleep 4

cd dashboard
npm install --silent
npm run dev &
DASHBOARD_PID=$!
cd ..

echo ""
echo "✅ ImmunoWatch is running!"
echo "   Dashboard:  http://localhost:3000"
echo "   API docs:   http://localhost:8000/docs"
echo "   Reports:    ./reports/"
echo ""
echo "💡 Click 'Simulate Infection' in the dashboard to see the AI system"
echo "   detect an infection event in real time."
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $API_PID $DASHBOARD_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
