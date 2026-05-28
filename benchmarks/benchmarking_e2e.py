import argparse
from cs336_basics.model import BasicsTransformerLM, annotated_scaled_dot_product_attention
from cs336_basics.optimizer import AdamW
import torch
from timeit import default_timer as timer
from tqdm import tqdm
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

import cs336_basics.model
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def main():
    torch.set_float32_matmul_precision('high')
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, choices=MODEL_CONFIGS.keys(), required=True, help='Model size to benchmark')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_warmup', type=int, default=0, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of training steps to benchmark')
    parser.add_argument('--mode', type=str, choices=['forward_only', 'forward_backward', 'full'], default='forward_only', help='Whether to benchmark forward pass or backward pass')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='float32', help='Data type for model parameters and computations')
    parser.add_argument('--layer_type', type=str, default='standard', choices=['standard', 'optimized'], help="Which layer implementation to benchmark")
    parser.add_argument('--profile_memory', action='store_true', help='Whether to profile memory usage with Nsight Systems')
    parser.add_argument('--compiled', action='store_true', help='Whether to use torch.compile')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = MODEL_CONFIGS[args.size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        context_length=args.context_length,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    if args.compiled:
        print(">>> Using torch.compile() for end-to-end benchmarking...")
        if args.num_warmup < 3:
            print("WARNING: num_warmup is too low for torch.compile. Forcing num_warmup=5 to complete compilation.")
            args.num_warmup = 5
        model = torch.compile(model, backend="inductor")
    
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
    
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.dtype == 'bfloat16' else nullcontext()
    
    # Warmup steps
    for _ in tqdm(range(args.num_warmup), desc="Warmup"):
        optimizer.zero_grad()
        
        torch.cuda.reset_peak_memory_stats()
        
        with ctx:
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        print(f"Peak GPU Memory: {peak_memory_bytes / (1024**2):.2f} MiB")
        
        optimizer.step()
        
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()
    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    step_times = []
    for _ in tqdm(range(args.num_steps), desc="Benchmark"):
        start_time = timer()
        
        if not args.profile_memory:
            torch.cuda.reset_peak_memory_stats()

        with nvtx.range("full_step"):
            with nvtx.range("forward_pass"):
                with ctx:
                    outputs = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, args.vocab_size), targets.view(-1))
            with nvtx.range("backward_pass"):
                if args.mode in ['forward_backward', 'full']:
                    loss.backward()
            
            if not args.profile_memory:
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                print(f"Peak GPU Memory: {peak_memory_bytes / (1024**2):.2f} MiB")
            
            with nvtx.range("optimizer_step"):
                if args.mode == 'full':
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end_time = timer()
        step_times.append(end_time - start_time)
        
    torch.cuda.cudart().cudaProfilerStop()
    if args.profile_memory:
        torch.cuda.memory._dump_snapshot("profiles/memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        
    avg_time = sum(step_times) / len(step_times)
    std = (sum((t - avg_time) ** 2 for t in step_times) / len(step_times)) ** 0.5
    print(f"Average time per step: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std:.4f} seconds")

if __name__ == "__main__":
    main()