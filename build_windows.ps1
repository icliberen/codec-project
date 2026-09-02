[CmdletBinding()]
param([string]$PythonExe = 'python', [switch]$SkipInstall)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
try {
  & $PythonExe -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Use Python 3.12 x64'"
  if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python interpreter.' }
  if (-not $SkipInstall) {
    & $PythonExe -m pip install -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw 'Build dependency installation failed.' }
  }
  & $PythonExe scripts/download_models.py
  if ($LASTEXITCODE -ne 0) { throw 'Model verification failed.' }
  & $PythonExe scripts/collect_licenses.py
  if ($LASTEXITCODE -ne 0) { throw 'License collection failed.' }
  & $PythonExe -m PyInstaller --noconfirm --clean --windowed --name 'Smart Codec' `
    --collect-all torch --collect-all ultralytics --collect-all tkinterdnd2 `
    --add-data 'assets;assets' --add-data 'LICENSE;.' --add-data 'THIRD_PARTY_NOTICES.txt;.' `
    --add-data 'third_party_licenses;third_party_licenses' `
    --add-data 'yolo11m-seg.pt;.' --add-data 'yolo11n.pt;.' start_gui.pyw
  if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
  Copy-Item -LiteralPath LICENSE,THIRD_PARTY_NOTICES.txt -Destination 'dist/Smart Codec'
  & "$PSScriptRoot/validate_windows_package.ps1" -MinimalEnvironment -StartupSeconds 30
} finally {
  Pop-Location
}
