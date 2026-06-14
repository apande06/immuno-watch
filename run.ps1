# ImmunoWatch — Windows launcher (PowerShell)
# Mirrors run.sh: generate data -> train -> evaluate -> start API + dashboard.

$ErrorActionPreference = "Stop"

Write-Host "ImmunoWatch — AI Health Monitoring System" -ForegroundColor Cyan
Write-Host "============================================="

# Resolve a real Python interpreter (avoid the Windows Store alias stub).
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py -or $py -like "*WindowsApps*") {
    $candidate = Get-Command py -ErrorAction SilentlyContinue
    if ($candidate) { $py = "py" } else { throw "Python 3 required" }
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "Node.js required" }

Write-Host "`nStep 1/5: Generating patient biosignal data..." -ForegroundColor Yellow
& $py data/simulator.py

Write-Host "`nStep 2/5: Training personal baseline models (LSTM Autoencoder)..." -ForegroundColor Yellow
& $py ml/baseline.py

Write-Host "`nStep 3/5: Training infection risk predictor (Temporal Transformer)..." -ForegroundColor Yellow
& $py ml/predictor.py

Write-Host "`nStep 4/5: Running federated learning simulation..." -ForegroundColor Yellow
& $py ml/federated.py

Write-Host "`nGenerating evaluation report..." -ForegroundColor Yellow
& $py ml/evaluation.py

Write-Host "`nStep 5/5: Starting services..." -ForegroundColor Green
$api = Start-Process uvicorn -ArgumentList "api.main:app","--host","0.0.0.0","--port","8000","--reload" -PassThru
Start-Sleep -Seconds 4

Push-Location dashboard
npm install --silent
$dash = Start-Process npm -ArgumentList "run","dev" -PassThru
Pop-Location

Write-Host "`nImmunoWatch is running!" -ForegroundColor Green
Write-Host "   Dashboard:  http://localhost:3000"
Write-Host "   API docs:   http://localhost:8000/docs"
Write-Host "   Reports:    ./reports/"
Write-Host "`nPress Enter to stop all services."
[void](Read-Host)

Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
Stop-Process -Id $dash.Id -ErrorAction SilentlyContinue
Write-Host "Stopped."
