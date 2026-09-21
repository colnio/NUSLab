param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'NUSLab\TelegramMCP'),
    [string]$Python = (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$sourceRoot = $PSScriptRoot
$installPath = [System.IO.Path]::GetFullPath($InstallRoot)
$packagePath = Join-Path $installPath 'TelegramNotifier'
$runtimePython = Join-Path $installPath '.venv\Scripts\python.exe'
$codexCommand = (Get-Command codex -ErrorAction Stop).Source
$configSource = Join-Path $sourceRoot 'config.local.json'
if (-not (Test-Path -LiteralPath $configSource)) {
    throw 'Link the local notifier before installing: python -m TelegramNotifier link'
}

New-Item -ItemType Directory -Path $packagePath -Force | Out-Null
foreach ($name in @('__init__.py', '__main__.py', 'notifier.py', 'mcp_server.py', 'run_mcp.py',
                     'requirements.txt', 'requirements-mcp.txt', 'README.md')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination (Join-Path $packagePath $name) -Force
}
# Preserve the installed credentials when updating the implementation.
$installedConfig = Join-Path $packagePath 'config.local.json'
if (-not (Test-Path -LiteralPath $installedConfig)) {
    Copy-Item -LiteralPath $configSource -Destination $installedConfig
}
if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $Python -m venv (Join-Path $installPath '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Telegram MCP Python environment.' }
}
# Use pip's bundled CA certificates; Windows trust-store enumeration can stall on this lab PC.
& $runtimePython -m pip install --disable-pip-version-check --use-deprecated=legacy-certs -r (Join-Path $packagePath 'requirements-mcp.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the Telegram MCP dependencies.' }
& $runtimePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Telegram MCP dependency verification failed.' }

$codexDirectory = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
$codexConfig = Join-Path $codexDirectory 'config.toml'
if (Test-Path -LiteralPath $codexConfig) {
    $backupPath = $codexConfig + '.before-telegram-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.bak'
    Copy-Item -LiteralPath $codexConfig -Destination $backupPath
}
& $codexCommand mcp add telegram -- $runtimePython (Join-Path $packagePath 'run_mcp.py')
if ($LASTEXITCODE -ne 0) { throw 'Could not register the user-wide Telegram MCP server.' }

# Allow a full upload request plus protocol overhead; preserve other config tables.
@'
import re
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
section = re.search(r'(?m)^\[mcp_servers\.telegram\]\s*$', text)
if section is None:
    raise SystemExit('Registered Telegram MCP table was not found.')
following = re.search(r'(?m)^\[', text[section.end():])
end = section.end() + following.start() if following else len(text)
body = text[section.end():end]
for name, value in [('startup_timeout_sec', 30), ('tool_timeout_sec', 120)]:
    body = re.sub(r'(?m)^' + name + r'\s*=.*\n?', '', body)
    body = '\n' + name + ' = ' + str(value) + '\n' + body.lstrip('\n')
updated = text[:section.end()] + body + text[end:]
tomllib.loads(updated)
path.write_text(updated, encoding='utf-8')
'@ | & $runtimePython - $codexConfig
if ($LASTEXITCODE -ne 0) { throw 'Could not configure Telegram MCP timeouts.' }
& $codexCommand mcp get telegram
if ($LASTEXITCODE -ne 0) { throw 'Telegram MCP registration verification failed.' }
Write-Host 'Telegram MCP is registered for all Codex projects for this Windows user. Restart the Codex client to load it.'
