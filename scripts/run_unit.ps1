# Build and run all unit tests via test/Makefile.
# Usage: .\scripts\run_unit.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

make -C test test
exit $LASTEXITCODE
