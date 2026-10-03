$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $Python)) {
    py -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked { & $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt") }
Invoke-Checked { & $Python -m pytest -q }
Invoke-Checked { & $Python -m src.experiment --run }

Write-Host "Full reproduction completed successfully."
