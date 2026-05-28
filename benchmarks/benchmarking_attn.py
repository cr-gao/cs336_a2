from cs336_basics.model import annotated_scaled_dot_product_attention
import torch
from timeit import default_timer as timer
from tqdm import tqdm
import torch.cuda.nvtx as nvtx
import gc

import cs336_basics.model
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def main():
    d_model = [16, 32, 64]
    seq_len = [256, 1024, 4096]
    batch_size = 8
    
    results = {}
    
    for d in d_model:
        for s in seq_len:
            print(f"Benchmarking d_model={d}, seq_len={s}")
            
            # Uncompiled version
            
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                q = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                k = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                v = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                
                # Warmup
                for _ in range(10):
                    out = annotated_scaled_dot_product_attention(q, k, v)
                    loss = out.sum()
                    loss.backward()
                    q.grad = k.grad = v.grad = None
                
                torch.cuda.synchronize()
                
                uncompiled_f_times = []
                for _ in tqdm(range(100)):
                    nvtx.range_push("Attention Forward")
                    start_time = timer()
                    
                    output = annotated_scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    
                    end_time = timer()
                    nvtx.range_pop()
                    uncompiled_f_times.append(end_time - start_time)
                    
                    del output

                avg_uncompiled_f = sum(uncompiled_f_times) / len(uncompiled_f_times)
                print(f"Average uncompiled forward pass time: {avg_uncompiled_f:.6f} seconds")
                
                output = annotated_scaled_dot_product_attention(q, k, v)
                loss = output.sum()
                torch.cuda.synchronize()
                
                # Memory in use
                uncompiled_memory_in_use = torch.cuda.memory_allocated() / (1024**2)
                print(f"Memory in use before backward pass: {uncompiled_memory_in_use:.2f} MB")
                
                loss.backward()
                q_grad = k.grad = v.grad = None
                del output, loss
                torch.cuda.synchronize()
                uncompiled_b_times = []
                for _ in tqdm(range(100)):
                    output = annotated_scaled_dot_product_attention(q, k, v)
                    loss = output.sum()
                    torch.cuda.synchronize()
                    
                    nvtx.range_push("Attention Backward")
                    start_time = timer()
                    
                    loss.backward()
                    torch.cuda.synchronize()
                    
                    end_time = timer() 
                    nvtx.range_pop()
                    uncompiled_b_times.append(end_time - start_time)
                    
                    q_grad = k.grad = v.grad = None
                    del output, loss
                    
                avg_uncompiled_b = sum(uncompiled_b_times) / len(uncompiled_b_times)
                print(f"Average uncompiled backward pass time: {avg_uncompiled_b:.6f} seconds")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    avg_uncompiled_f = avg_uncompiled_b = "OOM"
                    torch.cuda.empty_cache()
                else: raise e
                
            # Compiled version
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                q = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                k = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                v = torch.randn(batch_size, s, d, device='cuda', requires_grad=True)
                
                compiled_attention = torch.compile(annotated_scaled_dot_product_attention, backend="inductor")
                
                for _ in range(10):
                    out = compiled_attention(q, k, v)
                    loss = out.sum()
                    loss.backward()
                    q.grad = k.grad = v.grad = None
                    
                torch.cuda.synchronize()
                
                compiled_f_times = []
                for _ in tqdm(range(100)):
                    nvtx.range_push("Compiled Attention Forward")
                    start_time = timer()
                    
                    output = compiled_attention(q, k, v)
                    torch.cuda.synchronize()
                    
                    end_time = timer()
                    nvtx.range_pop()
                    compiled_f_times.append(end_time - start_time)
                    
                    del output
                    
                avg_compiled_f = sum(compiled_f_times) / len(compiled_f_times)
                print(f"Average compiled forward pass time: {avg_compiled_f:.6f} seconds")
                
                output = compiled_attention(q, k, v)
                loss = output.sum()
                torch.cuda.synchronize()
                
                compiled_memory_in_use = torch.cuda.memory_allocated() / (1024**2)
                print(f"Memory in use before backward pass: {compiled_memory_in_use:.2f} MB")
                loss.backward()
                q_grad = k.grad = v.grad = None
                del output, loss
                torch.cuda.synchronize()
                
                compiled_b_times = []
                for _ in tqdm(range(100)):
                    output = compiled_attention(q, k, v)
                    loss = output.sum()
                    torch.cuda.synchronize()
                    
                    nvtx.range_push("Compiled Backward")
                    start_time = timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    nvtx.range_pop()
                    compiled_b_times.append(timer() - start_time)
                    
                    q.grad = k.grad = v.grad = None
                    del output, loss
                    
                avg_compiled_b = sum(compiled_b_times) / len(compiled_b_times)
                print(f"Average compiled backward pass time: {avg_compiled_b:.6f} seconds")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    avg_compiled_f = avg_compiled_b = "OOM"
                    torch.cuda.empty_cache()
                else: raise e
                
            results[(d, s)] = {
                "uncompiled_forward": avg_uncompiled_f,
                "uncompiled_backward": avg_uncompiled_b,
                "compiled_forward": avg_compiled_f,
                "compiled_backward": avg_compiled_b,
                "uncompiled_memory_in_use": uncompiled_memory_in_use,
                "compiled_memory_in_use": compiled_memory_in_use
            }
                
    print("\n" + "#"*70 + "\nFINAL COMPILATION COMPARISON TABLE\n" + "#"*70)
    header = f"{'d_model':<8}{'seq_len':<8}{'Uncomp_F (s)':<14}{'Comp_F (s)':<12}{'Uncomp_B (s)':<14}{'Comp_B (s)':<12}"
    print(header)
    print("-" * len(header))
    
    for d in d_model:
        for s in seq_len:
            res = results[(d, s)]
            
            # 格式化输出，处理 OOM 字符串
            uf = f"{res['uncompiled_forward']:.5f}" if isinstance(res['uncompiled_forward'], float) else res['uncompiled_forward']
            cf = f"{res['compiled_forward']:.5f}" if isinstance(res['compiled_forward'], float) else res['compiled_forward']
            ub = f"{res['uncompiled_backward']:.5f}" if isinstance(res['uncompiled_backward'], float) else res['uncompiled_backward']
            cb = f"{res['compiled_backward']:.5f}" if isinstance(res['compiled_backward'], float) else res['compiled_backward']
            um = f"{res['uncompiled_memory_in_use']:.2f} MB" if isinstance(res['uncompiled_memory_in_use'], float) else res['uncompiled_memory_in_use']
            cm = f"{res['compiled_memory_in_use']:.2f} MB" if isinstance(res['compiled_memory_in_use'], float) else res['compiled_memory_in_use']
            
            print(f"{d:<8}{s:<8}{uf:<14}{cf:<12}{ub:<14}{cb:<12}{um:<20}{cm:<20}")

if __name__ == "__main__":
    main()