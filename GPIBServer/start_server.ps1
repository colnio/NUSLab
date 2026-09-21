param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Server environment is missing. Run GPIBServer\install_autostart.ps1 first."
}

$logDir = Join-Path $env:LOCALAPPDATA "NUSLab\GPIBServer"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Set-Location -LiteralPath $root
$ErrorActionPreference = "Continue"  # Uvicorn writes normal startup messages to stderr.
& $pythonPath -m uvicorn GPIBServer.app:app --host 127.0.0.1 --port 8765 --workers 1 2>&1 |
    Out-File -FilePath (Join-Path $logDir "server-console.log") -Append -Encoding utf8
exit $LASTEXITCODE
