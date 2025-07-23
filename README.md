# FNO Profiler Loop

A cleaned-up profiler loop for FNO (Fourier Neural Operator) models with memory profiling and timing capabilities.

## Features

- Small FNO model with configurable parameters
- Memory profiling with timeline export
- CUDA timing measurements
- Memory snapshot recording
- Command-line interface with Typer

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python profiler_loop_clean.py
```

### Custom Parameters

```bash
python profiler_loop_clean.py \
  --batch-size 16 \
  --in-channels 4 \
  --out-channels 2 \
  --hidden-channels 128 \
  --data-size 128 \
  --device cuda:0 \
  --save-dir ./my_profiler_outputs
```

### Available Parameters

- `--n-steps`: Number of total steps (default: 10)
- `--n-wait-steps`: Number of wait steps (default: 1)
- `--n-warmup-steps`: Number of warmup steps (default: 1)
- `--n-active-steps`: Number of active profiling steps (default: 3)
- `--n-repeats`: Number of profiling repeats (default: 2)
- `--batch-size`: Batch size (default: 8)
- `--in-channels`: Input channels (default: 3)
- `--out-channels`: Output channels (default: 1)
- `--hidden-channels`: Hidden channels (default: 64)
- `--data-size`: Data size (default: 64)
- `--device`: Device to use (default: cuda:0)
- `--save-dir`: Directory to save results (default: ./profiler_outputs)
- `--record-mem-snapshot`: Record memory snapshot (default: True)
- `--export-mem-timeline`: Export memory timeline (default: True)
- `--timing`: Enable timing measurements (default: True)
- `--nsight-profile`: Enable Nsight profiling (default: False)

## Output Files

The profiler generates several output files in the specified save directory:

- `max_mem.txt`: Memory breakdown at peak usage
- `memory_timeline.html`: Interactive memory timeline
- `memory_snapshot.pkl`: Memory snapshot data
- `cuda_timing.txt`: CUDA timing measurements

## Model Architecture

The FNO model is configured with:
- Input channels: Configurable via `--in-channels`
- Output channels: Configurable via `--out-channels`
- Hidden channels: Configurable via `--hidden-channels`
- Number of layers: Fixed at 4
- Optimizer: AdamW (no scheduler)
- Loss function: L2 loss

## Example Output

```
Starting profiling with FNO model:
  - Input channels: 3
  - Output channels: 1
  - Hidden channels: 64
  - Data size: 64x64
  - Batch size: 8
  - Device: cuda:0
### Resetting model and opt for timing...
Profiling completed. Results saved to ./profiler_outputs
``` 