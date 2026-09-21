param(
    [string]$PythonPath = "C:\Users\MeasurmentStand\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python was not found at $PythonPath"
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonPath -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create Python environment" }
}
& $venvPython -m pip install --no-deps --disable-pip-version-check --requirement (Join-Path $PSScriptRoot "requirements.lock")
if ($LASTEXITCODE -ne 0) { throw "Could not install server dependencies" }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Server dependencies failed validation" }

$userId = "$env:USERDOMAIN\$env:USERNAME"
$scriptPath = Join-Path $PSScriptRoot "start_server.ps1"
$powershellPath = Join-Path $PSHOME "powershell.exe"
$taskArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $scriptPath + '"'
$action = New-ScheduledTaskAction -Execute $powershellPath -Argument $taskArguments -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "NUSLab GPIB Server" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "NUSLab GPIB Server"
Write-Host "NUSLab GPIB Server task installed and started. Check http://127.0.0.1:8765/v1/health"
