# Multi-GPU training script using Accelerate (Windows PowerShell)
# Usage: .\origami_train_multigpu.ps1 -NumGpus 4 -Args @("--batch_size", "128", "--num_epochs", "1000")
# Or: .\origami_train_multigpu.ps1 4 --batch_size 128 --num_epochs 1000

param(
    [int]$NumGpus = 4,
    [string[]]$Args = @()
)

# Capture remaining positional args
if ($Args.Count -eq 0 -and $NumGpus -ne 4) {
    # If first arg looks like a number, use it; otherwise assume default 4 GPUs
    if ($NumGpus -is [int]) {
        $ActualNumGpus = $NumGpus
    } else {
        $ActualNumGpus = 4
    }
} else {
    $ActualNumGpus = $NumGpus
}

# Convert args to string if needed
$ArgsStr = if ($Args.Count -gt 0) { $Args -join " " } else { "" }

Write-Host "Launching training on $ActualNumGpus GPUs..." -ForegroundColor Green
Write-Host "Command: accelerate launch --config_file ../../accelerate_config.yaml --num_processes $ActualNumGpus origami_imitate_episodes.py $ArgsStr" -ForegroundColor Cyan
Write-Host ""

& accelerate launch --config_file ../../accelerate_config.yaml --num_processes $ActualNumGpus origami_imitate_episodes.py @Args
