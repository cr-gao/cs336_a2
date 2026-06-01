import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import pandas as pd

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))

def benchmark_allreduce(rank, world_size, tensor_size, queue):
    setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    tensor = torch.rand(tensor_size // 4, device=f"cuda:{rank}")
    
    # warm-up
    n_warmup = 5
    for _ in range(n_warmup):
        dist.all_reduce(tensor)      
    dist.barrier()
    
    # benchmark
    torch.cuda.synchronize(rank)
    start_time = time.perf_counter()
    
    n_iters = 10
    for _ in range(n_iters):
        dist.all_reduce(tensor)
    
    torch.cuda.synchronize(rank)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / n_iters
    
    local_avg_time = torch.tensor(avg_time, device=f"cuda:{rank}")
    dist.all_reduce(local_avg_time, op=dist.ReduceOp.MAX)
    
    if rank == 0:
        queue.put(local_avg_time.item())
    dist.destroy_process_group()
    

def main():
    tensor_sizes = [
        1 * 1024 * 1024,  # 1 MB
        10 * 1024 * 1024,  # 10 MB
        100 * 1024 * 1024,  # 100 MB
        1 * 1024 * 1024 * 1024  # 1 GB
    ]

    world_sizes = [
        2,
        4,
        6
    ]

    results = []
    
    ctx = mp.get_context("spawn")
    for world_size in world_sizes:
        print(f"At world Size: {world_size}")
        for tensor_size in tensor_sizes:
            print(f"At tensor size: {tensor_size / (1024*1024)} MB")
            queue = ctx.Queue()
            mp.spawn(benchmark_allreduce, args=(world_size, tensor_size, queue), nprocs=world_size)
            avg_time = queue.get()
            results.append({
                "world_size": world_size,
                "tensor_mb": tensor_size / (1024*1024),
                "avg_time_ms": avg_time * 1000
            })
            
    dataframe = pd.DataFrame(results)
    os.makedirs("profiles", exist_ok=True)
    dataframe.to_csv("profiles/benchmark_dcsn.csv", index=False)
    print(dataframe)
    
if __name__ == "__main__":
    main()