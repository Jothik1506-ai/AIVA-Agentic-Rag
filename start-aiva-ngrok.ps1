# ─────────────────────────────────────────────────────────────────────────────
# AIVA RAG – One-click launcher with ngrok public tunnel
#
# What this does:
#   1. Starts the Flask app (port 9072) in the background
#   2. Waits for it to be ready
#   3. Launches ngrok to expose it publicly
#   4. Prints the public URL for you to share
#
# Requirements:
#   - ngrok installed and on PATH  (https://ngrok.com/download)
#   - ngrok authtoken configured:  ngrok config add-authtoken <your_token>
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PORT = 9072
$PYTHON = ".\venv\Scripts\python.exe"

if (-not (Test-Path $PYTHON)) {
    Write-Host "ERROR: venv not found. Run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: ngrok not found. Download from https://ngrok.com/download and add to PATH." -ForegroundColor Red
    exit 1
}

# ── Start Flask ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Starting AIVA RAG server on port $PORT ..." -ForegroundColor Cyan
$flask = Start-Process -FilePath $PYTHON -ArgumentList "app.py" -PassThru -WindowStyle Normal

# Wait until Flask is accepting connections (max 30s)
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:$PORT/api/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        $ready = $true
        break
    } catch {}
}

if (-not $ready) {
    Write-Host "WARNING: Flask did not respond within 30s — starting ngrok anyway." -ForegroundColor Yellow
}

# ── Start ngrok ───────────────────────────────────────────────────────────────
Write-Host "  Launching ngrok tunnel ..." -ForegroundColor Cyan
$ngrok = Start-Process -FilePath "ngrok" -ArgumentList "http $PORT" -PassThru -WindowStyle Normal

Start-Sleep -Seconds 3

# Fetch public URL from ngrok local API
try {
    $tunnels = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -TimeoutSec 5
    $publicUrl = ($tunnels.tunnels | Where-Object { $_.proto -eq "https" }).public_url
    if (-not $publicUrl) {
        $publicUrl = $tunnels.tunnels[0].public_url
    }
} catch {
    $publicUrl = "(could not fetch — check ngrok window)"
}

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║  AIVA RAG is LIVE                                    ║" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║  Local :  http://localhost:$PORT                     ║" -ForegroundColor Green
Write-Host "  ║  Public:  $publicUrl" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║  Share the Public URL with your team.                ║" -ForegroundColor Green
Write-Host "  ║  When your PC turns OFF, Groq API takes over.        ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Press CTRL+C to stop everything." -ForegroundColor DarkGray

# Keep script alive so CTRL+C kills both processes
try {
    Wait-Process -Id $flask.Id
} finally {
    Write-Host "`n  Stopping ngrok..." -ForegroundColor Yellow
    Stop-Process -Id $ngrok.Id -ErrorAction SilentlyContinue
}
