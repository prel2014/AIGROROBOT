param(
    [Parameter(Mandatory)][string]$Salida,
    [ValidateSet('estacionada','roca','izquierda','derecha')]
    [string[]]$Tarea = @('estacionada','roca','izquierda','derecha'),
    [int]$Semilla = 751,
    [switch]$SoloFisica
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sim = if ($env:ISAAC_SIM_PATH) { $env:ISAAC_SIM_PATH } else { Join-Path $root '..\isaac-sim-6.2.0' }
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Salida)
[string[]]$extra = @()
if ($SoloFisica) { $extra += '--physics-only' }
foreach ($name in $Tarea) {
    $folder = Join-Path $destination $name
    if (Test-Path -LiteralPath $folder) { throw "Elige otra salida. Ya existe $folder" }
}
foreach ($name in $Tarea) {
    $folder = Join-Path $destination $name
    & (Join-Path $sim 'python.bat') (Join-Path $root 'tools\film_field_cycle.py') --viz none --crop citrus --orange-task $name --seed $Semilla --output $folder @extra
    if ($LASTEXITCODE -ne 0) { throw "Isaac Sim terminó con código $LASTEXITCODE en $name" }
    $report = Get-Content -LiteralPath (Join-Path $folder 'resultado.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "$name | éxito físico: $($report.robots[0].success) | $folder"
}
