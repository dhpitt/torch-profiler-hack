import sys
import os
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
from copy import deepcopy
from pickle import dump

import torch
from torch.cuda import Event as cudaEvent
import typer

from neuralop import H1Loss, LpLoss
from neuralop.models import FNO

from trace_handler import TimelineTracer

# Typer app for command line arguments
app = typer.Typer()

# Global parameters (all caps as requested)
N_STEPS = 10
N_WAIT_STEPS = 1
N_WARMUP_STEPS = 1
N_ACTIVE_STEPS = 3
N_REPEATS = 2
BATCH_SIZE = 8
IN_CHANNELS = 3
OUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
N_MODES = (32,32)
DATA_SIZE = 64
DEVICE = 'cuda:0'
SAVE_DIR = "./profiler_outputs"

@app.command()
def main(
    n_steps: int = N_STEPS,
    n_wait_steps: int = N_WAIT_STEPS,
    n_warmup_steps: int = N_WARMUP_STEPS,
    n_active_steps: int = N_ACTIVE_STEPS,
    n_repeats: int = N_REPEATS,
    batch_size: int = BATCH_SIZE,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    hidden_channels: int = HIDDEN_CHANNELS,
    n_modes: tuple[int, int] = N_MODES,
    data_size: int = DATA_SIZE,
    device: str = DEVICE,
    save_dir: str = SAVE_DIR,
    record_mem_snapshot: bool = True,
    export_mem_timeline: bool = True,
    timing: bool = True,
    nsight_profile: bool = False
):
    """
    Clean profiler loop for FNO model with memory profiling and timing.
    """
    
    # Set device
    device = torch.device(device)
    
    # Create small FNO model
    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=4
    )
    
    model = model.to(device)
    
    # Setup optimizer (AdamW, no scheduler)
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    train_loss = l2loss  # Use L2 loss for training
    
    # Memory recording setup
    if record_mem_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=100000)
    
    # Profiling setup
    NUM_BATCHES = n_repeats * (n_wait_steps + n_warmup_steps + n_active_steps) + 1
    
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    tracer_out_fpath = str(save_dir / "max_mem.txt")
    trace_handler = TimelineTracer(tracer_out_fpath)
    
    # Use full precision (no mixed precision)
    use_mixed_precision = False
    
    print(f"Starting profiling with FNO model:")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Output channels: {out_channels}")
    print(f"  - Hidden channels: {hidden_channels}")
    print(f"  - Data size: {data_size}x{data_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {device}")
    
    ##############
    # PROFILING  #
    ##############
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=n_wait_steps,
            warmup=n_warmup_steps,
            active=n_active_steps, 
            repeat=n_repeats
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        
        for idx in range(NUM_BATCHES):
            optimizer.zero_grad()
            
            # Benchmark with random data
            x = torch.randn(batch_size, in_channels, data_size, data_size).to(device)
            y = torch.randn(batch_size, out_channels, data_size, data_size).to(device)
            
            # Forward pass
            if use_mixed_precision:
                with torch.autocast("cuda"):
                    out = model(x)
                    loss = train_loss(out, y)
            else:
                out = model(x)
                loss = train_loss(out, y)
            
            loss.backward()
            optimizer.step()
            
            prof.step()
            
        if export_mem_timeline:
            timeline_export_fpath = str(save_dir / "memory_timeline.html")
            prof.export_memory_timeline(timeline_export_fpath)
    
    # Check unused parameters
    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    
    if record_mem_snapshot:
        snapshot = torch.cuda.memory._snapshot()
        snapshot_fpath = str(save_dir / "memory_snapshot.pkl")
        dump(snapshot, open(snapshot_fpath, 'wb'))
    
    ##########
    # TIMING #
    ##########
    if timing:
        event_totals = {
            'model_fwd': [],
            'loss_bk': [],
            'total_time': [],
            'opt_step': []
        }
        
        for _ in range(n_repeats):
            print(f"### Resetting model and opt for timing...")
            model = model.cpu()
            model = deepcopy(model)
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters())
            model.train()
            
            for idx in range(n_wait_steps + n_warmup_steps + n_active_steps):
                optimizer.zero_grad()
                
                x = torch.randn(batch_size, in_channels, data_size, data_size).to(device)
                y = torch.randn(batch_size, out_channels, data_size, data_size).to(device)
                
                # Time the forward call
                pre_fwd = cudaEvent(enable_timing=True)
                post_fwd = cudaEvent(enable_timing=True)
                
                pre_fwd.record()
                if use_mixed_precision:
                    with torch.autocast("cuda"):
                        out = model(x)
                else:
                    out = model(x)
                post_fwd.record()
                torch.cuda.synchronize()
                fwd_time = pre_fwd.elapsed_time(post_fwd)
                
                if use_mixed_precision:
                    with torch.autocast("cuda"):
                        loss = train_loss(out, y)
                else:
                    loss = train_loss(out, y)
                
                # Time the backpropagation
                pre_backward = cudaEvent(enable_timing=True)
                post_backward = cudaEvent(enable_timing=True)
                
                pre_backward.record()
                loss.backward()
                post_backward.record()
                
                torch.cuda.synchronize()
                backward_time = pre_backward.elapsed_time(post_backward)
                
                # Time optimizer step
                pre_opt_step = cudaEvent(enable_timing=True)
                post_opt_step = cudaEvent(enable_timing=True)
                
                pre_opt_step.record()
                optimizer.step()
                post_opt_step.record()
                torch.cuda.synchronize()
                opt_step_time = pre_opt_step.elapsed_time(post_opt_step)
                total_time = pre_fwd.elapsed_time(post_opt_step)
                
                # Only collate if we're past the first warmup steps
                if idx >= n_wait_steps + n_warmup_steps:
                    event_totals['model_fwd'].append(fwd_time)
                    event_totals['loss_bk'].append(backward_time)
                    event_totals['opt_step'].append(opt_step_time)
                    event_totals['total_time'].append(total_time)
        
        timing_msg = f"CUDA timing for key events:\n---------------\n"
        
        for key, times in event_totals.items():
            mean_time = np.mean(times)
            std = np.std(times)
            timing_msg += f"{key}: {mean_time:.3f}+/-{std:.3f}ms on CUDA\n"
        
        timing_fpath = str(save_dir / "cuda_timing.txt")
        with open(timing_fpath, "w") as f:
            f.write(timing_msg)
        f.close()
    
    print(f"Profiling completed. Results saved to {save_dir}")

if __name__ == "__main__":
    app() 