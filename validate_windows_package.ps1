[CmdletBinding()]
param(
  [string]$PackageDir,
  [ValidateRange(1, 60)]
  [int]$StartupSeconds = 8,
  [switch]$MinimalEnvironment
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($PackageDir)) {
  $PackageDir = Join-Path $PSScriptRoot 'dist\Smart Codec'
}

function Require-Path {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Description
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    throw "Missing ${Description}: $Path"
  }
}

$resolvedPackage = (Resolve-Path -LiteralPath $PackageDir).Path
$executablePath = Join-Path $resolvedPackage 'Smart Codec.exe'
$internalPath = Join-Path $resolvedPackage '_internal'

Require-Path -Path $executablePath -Description 'packaged executable'
Require-Path -Path $internalPath -Description 'PyInstaller runtime directory'

$executable = Get-Item -LiteralPath $executablePath
$internalEntries = @(Get-ChildItem -LiteralPath $internalPath -Force)
if ($internalEntries.Count -eq 0) {
  throw "PyInstaller runtime directory is empty: $internalPath"
}

$hash = (Get-FileHash -LiteralPath $executablePath -Algorithm SHA256).Hash
$startedAt = Get-Date
$packageProcess = $null
$startupPassed = $false
$mainWindowTitle = ''
$originalPath = $env:PATH

try {
  if ($MinimalEnvironment) {
    $systemRoot = [Environment]::GetEnvironmentVariable('SystemRoot')
    $env:PATH = Join-Path $systemRoot 'System32'
  }
  $packageProcess = Start-Process `
    -FilePath $executablePath `
    -WorkingDirectory $resolvedPackage `
    -WindowStyle Hidden `
    -PassThru

  $deadline = (Get-Date).AddSeconds($StartupSeconds)
  do {
    Start-Sleep -Milliseconds 250
    $packageProcess.Refresh()
    if ($packageProcess.HasExited) {
      throw "Packaged GUI exited during startup (exit code $($packageProcess.ExitCode))."
    }
    $mainWindowTitle = $packageProcess.MainWindowTitle
    if ($mainWindowTitle -like 'Unhandled exception*') {
      throw "Packaged GUI opened a Python error dialog: $mainWindowTitle"
    }
    if ($mainWindowTitle -like 'Smart Codec*') {
      $startupPassed = $true
      break
    }
  } while ((Get-Date) -lt $deadline)
  if (-not $startupPassed) {
    throw "Packaged GUI did not expose the Smart Codec main window within $StartupSeconds seconds. Last title: '$mainWindowTitle'"
  }
}
finally {
  if ($MinimalEnvironment) {
    $env:PATH = $originalPath
  }
  if ($null -ne $packageProcess) {
    $packageProcess.Refresh()
    if (-not $packageProcess.HasExited) {
      Stop-Process -Id $packageProcess.Id -Force
      $null = $packageProcess.WaitForExit(5000)
    }
  }
}

if (-not $startupPassed) {
  throw 'Packaged GUI startup smoke did not pass.'
}

[pscustomobject]@{
  status = 'passed'
  package = $resolvedPackage
  executable = $executable.Name
  executable_bytes = $executable.Length
  executable_sha256 = $hash
  internal_entries = $internalEntries.Count
  startup_seconds = $StartupSeconds
  minimal_environment = [bool]$MinimalEnvironment
  main_window_title = $mainWindowTitle
  started_at = $startedAt.ToString('o')
  checked_at = (Get-Date).ToString('o')
}
