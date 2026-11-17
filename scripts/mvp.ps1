param(
    [ValidateSet('doctor','teacher','evaluate-teacher','collect','split','train','evaluate-student','corrections','add-corrections')]
    [string]$Stage = 'doctor',
    [string]$Teacher,
    [string]$Student,
    [string]$TeacherEvaluation,
    [string]$Manifest,
    [string]$Dataset = 'datasets/demos',
    [string]$Scenario = 'red_cube',
    [int]$Robots = 4,
    [int]$Iterations = 3000,
    [int]$Episodes = 100,
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$BootstrapUpdates = -1,
    [ValidateSet(1.5,3,6,10,15)]
    [double]$StartLiftCm = 1.5,
    [ValidateRange(0.001,1)]
    [double]$ExplorationStdScale = 1,
    [switch]$Resume,
    [switch]$Headless,
    [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
)
$ErrorActionPreference = 'Stop'
if ($Resume -and $Stage -notin @('teacher','train')) { throw 'Resume solo aplica a teacher o train.' }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$' -or [Math]::Min($Robots,[Math]::Min($Iterations,[Math]::Min($Episodes,$Epochs))) -lt 1 -or $BatchSize -lt 1 -or $BootstrapUpdates -lt -1) {
    throw 'Usa un RunName sencillo y presupuestos positivos.'
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$config = Get-Content -Raw -LiteralPath (Join-Path $root 'configs/mvp.json') | ConvertFrom-Json
$runner = Join-Path $PSScriptRoot 'run.ps1'
[string[]]$visual = if ($Headless) { @('--headless') } else { @('--viz','kit') }
Push-Location $root
try {
    switch ($Stage) {
        'doctor' { & $runner doctor --output "logs/mvp/$RunName/doctor.json" }
        'teacher' {
            $bootstrap = if ($BootstrapUpdates -ge 0) { $BootstrapUpdates } else { $config.teacher_bootstrap_updates }
            [string[]]$resumeArgs = @()
            if ($Resume) {
                $resumeArgs = @('--resume', $(if ($Teacher) { $Teacher } else { 'latest' }))
            } elseif ($Teacher) { $resumeArgs = @('--checkpoint',$Teacher) }
            & $runner train_rl --task $config.task --num_envs $Robots --max_iterations $Iterations `
                --bootstrap-updates $bootstrap --bootstrap-batch-size $config.teacher_bootstrap_batch_size `
                --bootstrap-learning-rate $config.teacher_bootstrap_learning_rate `
                --start-lift-cm $StartLiftCm --exploration-std-scale $ExplorationStdScale `
                --scenario $Scenario --seed $config.training_seed --run_timestamp $RunName @resumeArgs @visual
        }
        'evaluate-teacher' {
            if (-not $Teacher) { throw 'Indica -Teacher con el checkpoint PPO.' }
            & $runner evaluate --scenario $Scenario --policy teacher --checkpoint $Teacher --num_envs $Robots `
                --episodes $config.episodes_per_seed --seeds @($config.evaluation_seeds) `
                --min-success-rate $config.min_success_rate --require-pass --output "logs/mvp/$RunName" @visual
        }
        'collect' {
            if (-not $Teacher -or -not $TeacherEvaluation) { throw 'Indica -Teacher y -TeacherEvaluation con evaluation.json aprobado.' }
            $gate = Get-Content -Raw -LiteralPath $TeacherEvaluation | ConvertFrom-Json
            $checkpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Teacher).Hash.ToLowerInvariant()
            if (-not $gate.passed -or -not $gate.complete -or $gate.policy -ne 'teacher' -or $gate.target_lift_m -ne $config.target_lift_m -or
                $gate.success_criterion -ne 'opposed_contact_minimum_lift_and_hold_v2' -or $gate.minimum_lift_required_m -lt $config.target_lift_m -or
                $gate.hold_seconds -lt 0.5 -or
                $gate.checkpoint_sha256 -ne $checkpointHash -or $gate.min_success_rate -lt $config.min_success_rate -or
                $gate.episodes_per_seed -lt $config.episodes_per_seed -or (Compare-Object @($gate.seeds) @($config.evaluation_seeds))) {
                throw 'La evaluacion no cumple el protocolo MVP para este checkpoint.'
            }
            & $runner collect_demos --scenario $Scenario --checkpoint $Teacher --episodes $Episodes --max-attempts ($Episodes * 10) `
                --num_envs $Robots --seed $config.collection_seed --output "datasets/demos/$RunName" @visual
        }
        'split' { & $runner dataset split --dataset $Dataset --seed $config.split_seed --output "datasets/manifests/$RunName.json" }
        'train' {
            if (-not $Manifest -and -not $Resume) { throw 'Indica -Manifest para mantener train/validation/test fijos.' }
            [string[]]$sourceArgs = @()
            if ($Resume) {
                $sourceArgs = @('--resume', $(if ($Student) { $Student } else { 'latest' }))
            } elseif ($Student) { $sourceArgs = @('--initialize-from',$Student) }
            [string[]]$manifestArgs = if ($Manifest) { @('--manifest',$Manifest) } else { @() }
            [string[]]$monitorArgs = if ($Headless) { @('--headless') } else { @('--live-plot') }
            & $runner train_vla @manifestArgs --output "checkpoints/vla/$RunName" `
                --sequence-length $config.student_sequence_length `
                --epochs $Epochs --batch-size $BatchSize --seed $config.training_seed @monitorArgs @sourceArgs
        }
        'evaluate-student' {
            if (-not $Student) { throw 'Indica -Student con best.pt.' }
            & $runner evaluate --scenario $Scenario --policy student --checkpoint $Student --num_envs $Robots `
                --episodes $config.episodes_per_seed --seeds @($config.evaluation_seeds) `
                --min-success-rate $config.min_success_rate --require-pass --output "logs/mvp/$RunName" @visual
        }
        'corrections' {
            if (-not $Teacher -or -not $Student) { throw 'Indica -Teacher y -Student.' }
            & $runner collect_corrections --scenario $Scenario --teacher $Teacher --student $Student --num_envs $Robots --episodes $Episodes `
                --seed $config.correction_seed --output "datasets/corrections/$RunName" @visual
        }
        'add-corrections' {
            if (-not $Manifest) { throw 'Indica -Manifest con las particiones originales.' }
            & $runner dataset add-corrections --manifest $Manifest --dataset $Dataset --output "datasets/manifests/$RunName.json"
        }
    }
} finally { Pop-Location }
