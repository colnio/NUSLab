param(
    [switch]$OneFile = $true
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

python .\generate_icons.py

$distRoot = Join-Path $PSScriptRoot "dist"
$buildRoot = Join-Path $PSScriptRoot "build"
$specRoot = Join-Path $PSScriptRoot "spec"
New-Item -ItemType Directory -Force -Path $distRoot | Out-Null
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $specRoot | Out-Null

$apps = @(
    @{ Script = "VAC.py"; Name = "Keithley-VAC"; Icon = "icons\VAC.ico" },
    @{ Script = "AVC.py"; Name = "Keithley-AVC"; Icon = "icons\AVC.ico" },
    @{ Script = "Pulses.py"; Name = "Keithley-Pulses"; Icon = "icons\PUL.ico" },
    @{ Script = "FET.py"; Name = "Keithley-FET"; Icon = "icons\FET.ico" },
    @{ Script = "FET_map.py"; Name = "Keithley-FET-Map"; Icon = "icons\MAP.ico" },
    @{ Script = "FourProbeResistance.py"; Name = "Keithley-4Probe-Resistance"; Icon = "icons\4PR.ico" },
    @{ Script = "FourProbeFET.py"; Name = "Keithley-4Probe-FET"; Icon = "icons\4PF.ico" }
)

$modeArgs = @()
if ($OneFile) {
    $modeArgs += "--onefile"
}

foreach ($app in $apps) {
    Write-Host "Building $($app.Name) from $($app.Script)..."
    $iconPath = (Resolve-Path (Join-Path $PSScriptRoot $app.Icon)).Path
    $args = @(
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name", $app.Name,
        "--icon", $iconPath,
        "--distpath", $distRoot,
        "--workpath", $buildRoot,
        "--specpath", $specRoot,
        "--collect-all", "pyqtgraph",
        "--collect-all", "matplotlib",
        "--collect-all", "pandas",
        "--collect-all", "numpy",
        "--hidden-import", "PyQt5.sip",
        $app.Script
    ) + $modeArgs
    pyinstaller @args
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed for $($app.Name) with exit code $LASTEXITCODE"
    }
}

Write-Host "Build complete. EXEs are in $distRoot"
