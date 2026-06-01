import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import pandas as pd
import os
import argparse
from contextlib import nullcontext
from tqdm import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP

MODEL_CONFIGS = {
        "small": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        "xl": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
        "10B": {
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36,
        },
    }

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)

def benchmark_worker(rank, world_size, args):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    config = MODEL_CONFIGS[args.size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        context_length=args.context_length,
    ).to(device)
    model = DDP(model)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
    
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.dtype == 'bfloat16' else nullcontext()
    
    # Warmup steps
    for _ in tqdm(range(args.num_warmup)):
        with ctx:
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        model.sync_gradients(optimizer)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()
        
    # Benchmarking steps
    step_times = []
    comm_times = []
    for _ in tqdm(range(args.num_steps)):
        torch.cuda.synchronize()
        step_start = time.time()
        
        with ctx:
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        
        torch.cuda.synchronize()
        comm_start = time.time()
        model.sync_gradients(optimizer)
        torch.cuda.synchronize()
        comm_end = time.time()
        
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        step_end = time.time()
        
        step_times.append(step_end - step_start)
        comm_times.append(comm_end - comm_start)
        
    avg_step_time = sum(step_times) / len(step_times)
    avg_comm_time = sum(comm_times) / len(comm_times)
    print(f"Average communication time for gradient synchronization on GPU {rank}: {avg_comm_time:.4f} seconds")
    print(f"Average step time on GPU {rank}: {avg_step_time:.4f} seconds")
    print(f"Average per-step communication overhead on GPU {rank}: {avg_comm_time / avg_step_time * 100:.2f}%")
    
    dist.destroy_process_group()
    
def main():
    world_size = 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, choices=MODEL_CONFIGS.keys(), required=True, help='Model size to benchmark')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_warmup', type=int, default=0, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of training steps to benchmark')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='float32', help='Data type for model parameters and computations')
    args = parser.parse_args()
    
    mp.spawn(benchmark_worker, args=(world_size, args), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    main()
    