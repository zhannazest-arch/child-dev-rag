# run.ps1 — однокомандный запуск проекта
#
# Запускает:
#   1. mcp_server.py — в отдельном окне PowerShell (stdio MCP-сервер)
#   2. app.py        — Gradio UI в текущем окне
#
# Использование:
#   .\run.ps1

$PROJECT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $PROJECT

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Child Development RAG — запуск проекта" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# ── MCP-сервер в отдельном окне ──────────────────────────────────────────────
Write-Host "[1/2] Запуск MCP-сервера в отдельном окне..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:PYTHONIOENCODING='utf-8'; Set-Location '$PROJECT'; Write-Host 'MCP Server running...' -ForegroundColor Green; python mcp_server.py"
)

Start-Sleep -Seconds 2

# ── Gradio UI в текущем окне ─────────────────────────────────────────────────
Write-Host "[2/2] Запуск Gradio UI (app.py)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Gradio откроется на http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "  Для остановки нажмите Ctrl+C" -ForegroundColor Gray
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
python app.py
