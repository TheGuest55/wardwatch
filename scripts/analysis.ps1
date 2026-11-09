#!/usr/bin/env pwsh
# PowerShell equivalent of scripts/analysis.sh — run a Task 4-5 analysis step inside the pinned
# wardwatch-analysis image. Uses `docker run` (never compose), mounts the repo at /work and mounts
# data/raw READ-ONLY so captured evidence can never be modified. Builds the image on first use.
#
# Usage (from the repo root, PowerShell):
#   .\scripts\analysis.ps1 features <session>
#   .\scripts\analysis.ps1 dataset   --split data\raw\sessions\split_v2.json --out-dir data\processed\v2 --force
#   .\scripts\analysis.ps1 datacheck --in-dir data\processed\v2 --out docs\data-report-v2.md
#   .\scripts\analysis.ps1 test
#   .\scripts\analysis.ps1 shell
#
# .ps1 blocked by execution policy? Run once per session:  Set-ExecutionPolicy -Scope Process Bypass
# Override the image with $env:ANALYSIS_IMAGE; the CPU cap with $env:ANALYSIS_CPUS.
$ErrorActionPreference = "Stop"

$Image = if ($env:ANALYSIS_IMAGE) { $env:ANALYSIS_IMAGE } else { "wardwatch-analysis:0.2.0" }
$Cpus  = if ($env:ANALYSIS_CPUS)  { $env:ANALYSIS_CPUS }  else { "1" }
# Repo root, forward-slashed so Docker Desktop accepts the volume paths (C:/Users/.../wardwatch).
$Repo  = ((Resolve-Path (Join-Path $PSScriptRoot "..")).Path) -replace '\\', '/'

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[analysis] building $Image (first run only)..."
    docker build -t $Image "$Repo/services/analysis"
}

function Invoke-Analysis {
    param([string[]] $CmdArgs)
    docker run --rm --network none --cpus $Cpus `
        -v "${Repo}:/work" -v "${Repo}/data/raw:/work/data/raw:ro" -w /work `
        $Image @CmdArgs
}

$cmd  = if ($args.Count -ge 1) { $args[0] } else { "" }
$rest = if ($args.Count -ge 2) { $args[1..($args.Count - 1)] } else { @() }

switch ($cmd) {
    "features" {
        if ($rest.Count -lt 1) { throw "usage: analysis.ps1 features <session>" }
        Invoke-Analysis (@("python", "-m", "src.features.extract", "--session") + $rest)
    }
    "dataset"   { Invoke-Analysis (@("python", "-m", "src.dataset") + $rest) }
    "datacheck" { Invoke-Analysis (@("python", "-m", "src.datacheck.check") + $rest) }
    "test"      { Invoke-Analysis @("python", "-m", "unittest", "discover", "-s", "tests/pipeline", "-p", "test_*.py") }
    "shell"     { Invoke-Analysis @("bash") }
    default     { throw "usage: analysis.ps1 {features <session>|dataset|datacheck|test|shell}" }
}
