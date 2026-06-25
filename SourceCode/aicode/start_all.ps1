# start_all.ps1
# Run all three services for the FruitAI application
# Usage: .\start_all.ps1

Write-Host "Starting FruitAI Application..." -ForegroundColor Green
Write-Host ""

# AI Service
Write-Host "[1/3] Starting Python AI Service (port 5001)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd d:\aicode\ai_service; & 'C:/Users/OS/anaconda3/envs/tdenv/python.exe' app.py`"" -WindowStyle Normal

Start-Sleep -Seconds 2

# Backend
Write-Host "[2/3] Starting Node.js Backend (port 5000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd d:\aicode\backend; npm run dev`"" -WindowStyle Normal

Start-Sleep -Seconds 2

# Frontend
Write-Host "[3/3] Starting React Frontend (port 5173)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd d:\aicode\frontend; npm run dev`"" -WindowStyle Normal

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host "Backend:  http://localhost:5000" -ForegroundColor Yellow
Write-Host "AI API:   http://localhost:5001" -ForegroundColor Yellow
