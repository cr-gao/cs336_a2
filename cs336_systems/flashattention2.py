import torch
import triton
import triton.language as tl

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        b_q = 16
        b_k = 16
        b_v = 16
        
        t_q = torch.ceil(torch.tensor(Q.shape[-2]) / torch.tensor(b_q)).long()
        t_k = torch.ceil(torch.tensor(K.shape[-2]) / torch.tensor(b_k)).long()
        
        O = []
        L = []
        
        for i in range(t_q):
            q_i = Q[:, i*b_q: (i+1)*b_q, :]
            o_i_prev = torch.zeros_like(q_i)
            l_i_prev = torch.zeros((q_i.shape[0], q_i.shape[-2]), dtype=torch.float32, device=Q.device)
            m_i_prev = torch.full((q_i.shape[0], q_i.shape[-2]), float("-inf"), dtype=torch.float32, device=Q.device)
            
            for j in range(t_k):
                k_j = K[:, j*b_k: (j+1)*b_k, :]
                v_j = V[:, j*b_v: (j+1)*b_v, :]
                
                s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
                row_max = torch.max(s_ij, dim=-1)[0]
                m_ij = torch.max(row_max, m_i_prev)
                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))
                l_ij = torch.exp(m_i_prev - m_ij) * l_i_prev + torch.sum(p_ij, dim=-1)
                o_ij = torch.diag_embed(torch.exp(m_i_prev - m_ij)) @ o_i_prev + p_ij @ v_j
                
                l_i_prev = l_ij
                m_i_prev = m_ij
                o_i_prev = o_ij
            
            o_i = torch.inverse(torch.diag_embed(l_i_prev)) @ o_i_prev
            l_i = m_i_prev + torch.log(l_i_prev)
            
            O.append(o_i)
            L.append(l_i)
        O = torch.cat(O, dim=-2)
        L = torch.cat(L, dim=-1)
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal        
        return O
    
    @torch.compile
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        D = (O * dO).sum(dim=-1)
        S = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        
        if ctx.is_causal:
            query_indices = torch.arange(Q.shape[-2], device=Q.device)[:, None]
            key_indices = torch.arange(K.shape[-2], device=Q.device)[None, :]
            mask = query_indices < key_indices
            S = S.masked_fill(mask, float("-inf"))
            
        P = torch.exp(S - L.unsqueeze(-1))
        dV = torch.matmul(P.transpose(-2, -1), dO)
        dP = torch.matmul(dO, V.transpose(-2, -1))
        dS = P * (dP - D.unsqueeze(-1))
        dQ = torch.matmul(dS, K) / (Q.shape[-1] ** 0.5)
        dK = torch.matmul(dS.transpose(-2, -1), Q) / (Q.shape[-1] ** 0.5)
        
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, 
        shape = (N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    Q = tl.load(Q_block_ptr)
    out = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    
    for i in range(0, N_KEYS, K_TILE_SIZE):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)
        
        S_tile = tl.dot(Q, tl.trans(K_tile), allow_tf32=True) * scale

        if is_causal:
            query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_indices = i + tl.arange(0, K_TILE_SIZE)
            mask = query_indices[:, None] < key_indices[None, :]
            S_tile = tl.where(mask, float("-inf"), S_tile)

        row_max = tl.max(S_tile, axis=-1)
        m_tile = tl.maximum(m, row_max)
        P_tile = tl.exp(S_tile - m_tile[:, None]).to(tl.float32)
        alpha = tl.exp(m - m_tile).to(tl.float32)
        l = alpha * l + tl.sum(P_tile, axis=-1)
        out = alpha[:, None] * out + tl.dot(P_tile, V_tile)
        m = m_tile
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    O = out / l[:, None]
    L = m + tl.log(l)
    tl.store(O_block_ptr, O)
    tl.store(L_block_ptr, L)

class FlashAttention2_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, D = Q.shape
        
        O = torch.empty_like(Q)
        L = torch.empty((B, N), dtype=torch.float32, device=Q.device)
        
        scale = 1.0 / (D ** 0.5)
        
        grid = (triton.cdiv(N, 16), B)  # Assuming tile size of 16 for queries
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N, N,
            scale,
            D,
            16,  # Q_TILE_SIZE
            16,  # K_TILE_SIZE
            is_causal
        )
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal        
        return O
    
    @torch.compile
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        D = (O * dO).sum(dim=-1)
        S = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        
        if ctx.is_causal:
            query_indices = torch.arange(Q.shape[-2], device=Q.device)[:, None]
            key_indices = torch.arange(K.shape[-2], device=Q.device)[None, :]
            mask = query_indices < key_indices
            S = S.masked_fill(mask, float("-inf"))
            
        P = torch.exp(S - L.unsqueeze(-1))
        dV = torch.matmul(P.transpose(-2, -1), dO)
        dP = torch.matmul(dO, V.transpose(-2, -1))
        dS = P * (dP - D.unsqueeze(-1))
        dQ = torch.matmul(dS, K) / (Q.shape[-1] ** 0.5)
        dK = torch.matmul(dS.transpose(-2, -1), Q) / (Q.shape[-1] ** 0.5)
        
        return dQ, dK, dV, None
        
            
        
        
        
    
