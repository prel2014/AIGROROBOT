param(
    [ValidateSet('scene_gallery','preview','train_rl','play','collect_demos','train_vla','play_vla','infer_vla','memory_rl','evaluate','dataset','collect_corrections','doctor')]
    [string]$Command
)
$ErrorActionPreference = 'Stop'
if (-not $Command) { throw 'Indica un comando; consulta README.md o scripts/mvp.ps1.' }
$ExtraArgs = $args
# Isaac Lab 3 selects headless simulation with --viz none. Keep the public
# --headless option for this project's launchers; offline training owns its flag.
if ($Command -in @('scene_gallery','preview','train_rl','play','collect_demos','play_vla','memory_rl','evaluate','collect_corrections') -and
    $ExtraArgs -contains '--headless') {
    if (@($ExtraArgs | Where-Object { $_ -match '^--(viz|visualizer)(=|$)' }).Count) {
        throw 'Usa --headless o --viz; no combines las dos opciones.'
    }
    $ExtraArgs = @($ExtraArgs | Where-Object { $_ -ne '--headless' })
    $ExtraArgs += @('--viz', 'none')
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$simPath = if ($env:ISAAC_SIM_PATH) { $env:ISAAC_SIM_PATH } else { Join-Path $projectRoot '..\isaac-sim-6.2.0' }
if ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) { throw 'Desactiva conda/venv antes de usar el Python de Isaac Sim.' }
$env:DEXMAN_ROOT = $projectRoot
$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $projectRoot
try {
    [string[]]$checkArgs = @('--command', $Command)
    if ($ExtraArgs -contains '--live-plot') { $checkArgs += '--live-plot' }
    & (Join-Path $simPath 'python.bat') (Join-Path $PSScriptRoot 'check_runtime.py') @checkArgs
    if ($LASTEXITCODE -ne 0) { throw 'Entorno incompleto. Completa la instalacion indicada arriba y vuelve a ejecutar.' }
    & (Join-Path $simPath 'python.bat') (Join-Path $PSScriptRoot "$Command.py") @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "$Command fallo con codigo $LASTEXITCODE" }
} finally {
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
    Pop-Location
}
