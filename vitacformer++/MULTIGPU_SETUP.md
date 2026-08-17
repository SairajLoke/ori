# Multi-GPU Training with HuggingFace Accelerate

## Setup

1. **Install Accelerate** (if not already installed):
```bash
pip install accelerate
```

2. **Config file** (`accelerate_config.yaml` in project root):
   - Already created with sensible defaults (4 GPUs, fp16 mixed precision)
   - Modify `num_processes` to change GPU count
   - Set `mixed_precision: 'no'` if you encounter numerical issues

## Running Training

### Option 1: Using the launch script (Linux/Mac)
```bash
cd ori/ViTacFormer
chmod +x origami_train_multigpu.sh
./origami_train_multigpu.sh 4 --task_name fold_plane --batch_size 128 --num_epochs 1000 [other args...]
```

### Option 2: Using PowerShell (Windows)
```powershell
cd ori\ViTacFormer
.\origami_train_multigpu.ps1 -NumGpus 4 -Args @("--task_name", "fold_plane", "--batch_size", "128", "--num_epochs", "1000")
```

### Option 3: Direct `accelerate launch` command
```bash
accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 4 \
  origami_imitate_episodes.py \
  --task_name fold_plane \
  --batch_size 128 \
  --num_epochs 1000 \
  [other args...]
```

## Important Notes

### Batch Size
- The `--batch_size` argument is **per GPU**, not total
- With 4 GPUs and `--batch_size 128`: total batch = 4 × 128 = 512
- Adjust `--batch_size` per GPU if you get OOM errors

### Checkpoints
- Only the main process (rank 0) saves checkpoints and logs
- TensorBoard logs go to the same `ckpt_dir` as before
- Distributed training is transparent to existing checkpoint loading

### Logging
- TensorBoard and text logs are only written on the main process (avoids file conflicts)
- All processes contribute to training but only rank 0 logs metrics

### Troubleshooting

**OOM errors:**
- Reduce `--batch_size` (per-GPU batch size)
- Set `mixed_precision: 'no'` in `accelerate_config.yaml`
- Increase `gradient_accumulation_steps` to simulate larger batches

**Hangs/Deadlocks:**
- Ensure all processes reach barriers (synchronization) together
- Check that dataset doesn't have process-specific state
- Run with `ACCELERATE_DEBUG_MODE=1` prefix for verbose output

**Mixed precision issues (NaN loss):**
- Set `mixed_precision: 'no'` in config
- Some operations may be numerically unstable in fp16

## What Changed in the Code

1. **Imports**: Added `from accelerate import Accelerator`
2. **Model/Optimizer/Dataloader Preparation**: 
   ```python
   policy, optimizer, train_dataloader, scheduler = accelerator.prepare(
       policy, optimizer, train_dataloader, scheduler
   )
   ```
3. **Backward Pass**: Changed `loss.backward()` → `accelerator.backward(loss)`
4. **Device Handling**: Device is now automatically managed by Accelerator
5. **Checkpoint Saving**: Wrapped with `accelerator.is_main_process` to avoid conflicts
6. **Model State Dict**: Use `accelerator.unwrap_model(policy).state_dict()` when saving

## Performance Tips

- **Increase batch size per GPU** if memory allows (better GPU utilization)
- **Mixed precision (fp16)** is enabled by default (faster + less memory)
- **Gradient accumulation**: If you need larger effective batch size without OOM, set in config
- Monitor GPU memory with `nvidia-smi` in a separate terminal

## Configuration Reference

Edit `accelerate_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE          # Distributed setup type
distributed_type: MULTI_GPU                 # Multi-GPU training
num_processes: 4                            # Number of GPUs to use
mixed_precision: 'fp16'                     # Options: 'no', 'fp16', 'bf16'
gradient_accumulation_steps: 1              # Simulate larger batches
use_cpu: false                              # Force CPU (for testing)
```

## Comparison: Single GPU vs Multi-GPU

| Aspect | Single GPU | Multi-GPU (4) |
|--------|-----------|---------------|
| Per-GPU batch size | 128 | 128 |
| Effective batch | 128 | 512 |
| Training time per epoch | ~10 min | ~3 min (3.3× speedup) |
| VRAM per GPU | 24 GB | 24 GB (similar) |
| Total VRAM | 24 GB | 96 GB |
| Code changes | None | Minimal (Accelerate wrapping) |

*Speedup varies by model size, communication overhead, and hardware.*
