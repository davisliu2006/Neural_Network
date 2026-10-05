# Build (via test/Makefile) and run an integration test from the repo root.
# Usage: .\scripts\run_integration.ps1 <name>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

$Src = Join-Path $Root "test\integration\$Name.cpp"
if (-not (Test-Path $Src)) {
    Write-Error "integration test not found: test/integration/$Name.cpp"
}

Write-Host "compile: test/integration/$Name.cpp"
make -C test "bin/$Name.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Exe = Join-Path $Root "test\bin\$Name.exe"
Write-Host "run:     test/bin/$Name.exe"
& $Exe
exit $LASTEXITCODE
