param([string]$IsaacSimPath = (Join-Path $PSScriptRoot '..\..\isaac-sim-6.2.0'))
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$simPath = (Resolve-Path $IsaacSimPath).Path
$labPath = Join-Path (Split-Path -Parent $simPath) 'IsaacLab'
if ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
    throw 'Desactiva conda/venv: este proyecto usa el Python 3.12 incluido en Isaac Sim.'
}
if (-not (Test-Path -LiteralPath (Join-Path $simPath 'python.bat'))) { throw 'Isaac Sim no encontrado.' }
$labInstaller = Join-Path $labPath 'isaaclab.bat'
if (-not (Test-Path -LiteralPath $labInstaller -PathType Leaf)) {
    throw "Extrae en $labPath la revisión de Isaac Lab indicada en configs/runtime.lock.json."
}
$simLink = Join-Path $labPath '_isaac_sim'
if (-not (Test-Path -LiteralPath $simLink)) {
    # Junctions work on Windows without administrator privileges.
    New-Item -ItemType Junction -Path $simLink -Target $simPath | Out-Null
}
$env:ISAAC_SIM_PATH = $simPath
$env:DEXMAN_ROOT = $projectRoot
Push-Location $labPath
try {
    & $labInstaller -i
    if ($LASTEXITCODE -ne 0) { throw 'Fallo instalando Isaac Lab.' }
    & (Join-Path $simPath 'python.bat') -m pip install -c (Join-Path $projectRoot 'configs/runtime-constraints.txt') -e $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Fallo instalando dexman.' }
} finally { Pop-Location }
Write-Host 'Dependencias instaladas. No se ha iniciado ninguna simulación ni entrenamiento.'
