# Remove built test binaries via test/Makefile.
# Usage: .\scripts\clean.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

make -C test clean
exit $LASTEXITCODE
