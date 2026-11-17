param(
    [ValidateSet('preview','teacher','collect','train','evaluate','infer')]
    [string]$Stage = 'preview',
    [int]$Robots = 4,
    [switch]$Resume,
    [switch]$Headless,
    [string]$Checkpoint,
    [string]$Dataset = 'datasets/demos',
    [string]$Episode,
    [int]$Episodes = 100,
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$ExecuteSteps = 1,
    [int]$Horizon = 8,
    [int]$Frame = 0,
    [string]$MemoryIn,
    [string]$Instruction,
    [int]$Iterations = 3000,
    [int]$BootstrapUpdates = 600,
    [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
)
$ErrorActionPreference = 'Stop'
if ($Resume -and $Stage -notin @('teacher','train')) { throw 'Resume solo aplica a teacher o train.' }
if ($Robots -lt 1 -or $Episodes -lt 1 -or $Epochs -lt 1 -or $BatchSize -lt 1 -or $Iterations -lt 1 -or $BootstrapUpdates -lt 0 -or
    $ExecuteSteps -lt 1 -or $Horizon -lt 1 -or $Frame -lt 0) {
    throw 'Robots, Episodes, Epochs, BatchSize, Iterations, ExecuteSteps y Horizon deben ser positivos; Frame >= 0.'
}
if ($PSBoundParameters.ContainsKey('Instruction') -and [string]::IsNullOrWhiteSpace($Instruction)) {
    throw 'Instruction debe contener texto.'
}
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName solo admite letras, numeros, _ y -.' }
if ($Stage -in @('collect','evaluate','infer') -and -not $Checkpoint) {
    throw 'Esta etapa requiere -Checkpoint: PPO para collect; VLA para evaluate/infer.'
}
$runner = Join-Path $PSScriptRoot 'run.ps1'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
[string[]]$visualArgs = if ($Headless) { @('--headless') } else { @('--viz','kit') }
[string[]]$instructionArgs = if ($Instruction) { @('--instruction', $Instruction) } else { @() }
Push-Location $root
try {
    switch ($Stage) {
        'preview' {
            & $runner preview --task Dexman-UR10e-Shadow-Lift-RGBD-v1 --num_envs $Robots @visualArgs
        }
        'teacher' {
            Write-Host 'Teacher PPO: estado -> 26 mandos; genera demostraciones para la politica visual.'
            $runDirectory = Join-Path $root "logs/rsl_rl/ur10e_shadow_lift_v1/$RunName"
            if (Test-Path -LiteralPath $runDirectory) { throw 'RunName ya existe; elige otro nombre.' }
            [string[]]$resumeArgs = @()
            if ($Resume) {
                $resumeArgs = @('--resume', $(if ($Checkpoint) { $Checkpoint } else { 'latest' }))
            } elseif ($Checkpoint) { $resumeArgs = @('--checkpoint', $Checkpoint) }
            & $runner train_rl --task Dexman-UR10e-Shadow-Lift-v1 --rl_library rsl_rl --num_envs $Robots `
                --max_iterations $Iterations --bootstrap-updates $BootstrapUpdates --run_timestamp $RunName @resumeArgs @visualArgs
        }
        'collect' {
            & $runner collect_demos --checkpoint $Checkpoint --num_envs $Robots --episodes $Episodes `
                --max-attempts ([Math]::Max(1000, $Episodes * 10)) --output "datasets/demos/$RunName" @instructionArgs @visualArgs
        }
        'train' {
            Write-Host 'VLA v4: tokens espaciales CLIP, texto, q/dq y memoria -> angulos en radianes y memoria actualizada.'
            [string[]]$monitorArgs = if ($Headless) { @('--headless') } else { @('--live-plot') }
            [string[]]$sourceArgs = @()
            if ($Resume) {
                $sourceArgs = @('--resume', $(if ($Checkpoint) { $Checkpoint } else { 'latest' }))
            } elseif ($Checkpoint) { $sourceArgs = @('--initialize-from', $Checkpoint) }
            & $runner train_vla --dataset $Dataset --epochs $Epochs --batch-size $BatchSize --horizon $Horizon `
                --output "checkpoints/vla/$RunName" @monitorArgs @sourceArgs
        }
        'evaluate' {
            & $runner play_vla --checkpoint $Checkpoint --num_envs $Robots --episodes $Episodes `
                --execute-steps $ExecuteSteps --output "logs/vla/$RunName.json" @instructionArgs @visualArgs
        }
        'infer' {
            if (-not $Episode) { throw 'Infer requiere -Episode con un archivo HDF5 grabado.' }
            [string[]]$memoryArgs = if ($MemoryIn) { @('--memory-in', $MemoryIn) }
            & $runner infer_vla --checkpoint $Checkpoint --episode $Episode --frame $Frame `
                --output "logs/vla/$RunName.npz" @instructionArgs @memoryArgs
        }
    }
} finally { Pop-Location }
