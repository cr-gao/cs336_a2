import torch
import triton
import pandas as pd
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flashattention2 import FlashAttention2_Triton

def benchmark_forward(attn_fn, Q, K, V, mask=None, is_causal=False, is_flash_attn=False):
    return triton.testing.do_bench(
        lambda: attn_fn(Q, K, V, mask) if not is_flash_attn else attn_fn(Q, K, V, is_causal)
    )

def benchmark_backward(attn_fn, Q, K, V, mask=None, is_causal=False, is_flash_attn=False):
    O = attn_fn(Q, K, V, mask) if not is_flash_attn else attn_fn(Q, K, V, is_causal)

    dO = torch.randn_like(O)

    return triton.testing.do_bench(
        lambda: torch.autograd.grad(
            outputs=O,
            inputs=(Q,K,V),
            grad_outputs=dO,
            retain_graph=True,
        )
    )
    
def benchmark_e2e(attn_fn, Q, K, V, mask=None, is_causal=False, is_flash_attn=False):
    dO = None
    def run():
        O = attn_fn(Q, K, V, mask) if not is_flash_attn else attn_fn(Q, K, V, is_causal)

        nonlocal dO
        if dO is None:
            dO = torch.randn_like(O)

        torch.autograd.grad(
            outputs=O,
            inputs=(Q,K,V),
            grad_outputs=dO,
        )

    return triton.testing.do_bench(run)

def main():
    seq_lengths = [
        128,
        256,
        512,
    ]

    head_dims = [
        16,
        32
    ]

    dtypes = [
        torch.float32,
        torch.bfloat16
    ]

    batch_size = 1
    causal = True

    results = []
    for dtype in dtypes:
        for head_dim in head_dims:
            for seq_length in seq_lengths:
                Q = torch.randn(batch_size, seq_length, head_dim, device='cuda', dtype=dtype, requires_grad=True)
                K = torch.randn(batch_size, seq_length, head_dim, device='cuda', dtype=dtype, requires_grad=True)
                V = torch.randn(batch_size, seq_length, head_dim, device='cuda', dtype=dtype, requires_grad=True)

                mask = None
                if causal:
                    mask = torch.tril(torch.ones(seq_length, seq_length, device='cuda', dtype=torch.bool))
                    
                pytorch_attn_time = benchmark_forward(scaled_dot_product_attention, Q, K, V, mask, is_causal=False, is_flash_attn=False)
                pytorch_attn_bw_time = benchmark_backward(scaled_dot_product_attention, Q, K, V, mask, is_causal=False, is_flash_attn=False)
                pytorch_attn_e2e_time = benchmark_e2e(scaled_dot_product_attention, Q, K, V, mask, is_causal=False, is_flash_attn=False)

                flash_attn_time = benchmark_forward(FlashAttention2_Triton.apply, Q, K, V, is_causal=causal, is_flash_attn=True)
                flash_attn_bw_time = benchmark_backward(FlashAttention2_Triton.apply, Q, K, V, is_causal=causal, is_flash_attn=True)
                flash_attn_e2e_time = benchmark_e2e(FlashAttention2_Triton.apply, Q, K, V, is_causal=causal, is_flash_attn=True)

                results.append({
                    'dtype': dtype,
                    'head_dim': head_dim,
                    'seq_length': seq_length,
                    
                    'pytorch_attn_time': pytorch_attn_time,
                    'pytorch_attn_bw_time': pytorch_attn_bw_time,
                    'pytorch_attn_e2e_time': pytorch_attn_e2e_time,
                    
                    'flash_attn_time': flash_attn_time,
                    'flash_attn_bw_time': flash_attn_bw_time,
                    'flash_attn_e2e_time': flash_attn_e2e_time
                })
                
    dataframe = pd.DataFrame(results)
    dataframe.to_csv('profiles/flash_attn_benchmark_results.csv', index=False)
    print(dataframe)
    
if __name__ == "__main__":
    main()
        
