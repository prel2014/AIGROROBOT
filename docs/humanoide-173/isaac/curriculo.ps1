[CmdletBinding()]
param(
    [string]$PythonIsaac = "python.bat",
    [string]$PythonPlan = "python",
    [string]$Salida = "runs/habilidades",
    [ValidateRange(1,16)][int]$Robots = 16,
    [int]$Semilla = 17,
    [ValidateRange(1,1000000)][int]$Actualizaciones = 300,
    [switch]$Iniciar,
    [switch]$Reanudar
)
$ErrorActionPreference = "Stop"
$entrada = Join-Path $PSScriptRoot "entrenar_habilidades.py"
$matriz = & $PythonPlan $entrada plan --json
if ($LASTEXITCODE -ne 0) { throw "No se pudo leer el currículo." }
$planes = ($matriz | ConvertFrom-Json).PSObject.Properties
foreach ($plan in $planes) {
    foreach ($entorno in $plan.Value) {
        $anterior = $null
        foreach ($nivel in 0..2) {
            $perfil = Join-Path $Salida "$($plan.Name)/$entorno/nivel$nivel/semilla$Semilla/politica.pt"
            $argumentos = @($entrada, "entrenar", "--tarea", $plan.Name, "--entorno", $entorno,
                "--nivel", $nivel, "--robots", $Robots, "--semilla", $Semilla, "--salida", $Salida,
                "--actualizaciones", $Actualizaciones)
            if ($Reanudar -and (Test-Path -LiteralPath $perfil)) {
                $argumentos += "--reanudar"
            } elseif ($anterior) {
                $argumentos += @("--inicial", $anterior)
            }
            Write-Output "$($plan.Name) / $entorno / nivel $nivel / $perfil"
            if ($Iniciar) {
                & $PythonIsaac @argumentos
                if ($LASTEXITCODE -ne 0) { throw "Falló el perfil $perfil." }
            }
            $anterior = $perfil
        }
    }
}
