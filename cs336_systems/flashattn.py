import torch
import triton
import triton.language as tl
from einops import einsum

class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    def __init__(self):
        super().__init__() 

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B0, B1 = 16, 16
        b, s, d = Q.shape
        L = torch.empty((b, s,), device=Q.device, dtype=Q.dtype)
        O = torch.empty((b, s, d), device=Q.device, dtype=Q.dtype)
        for i in range(0, s, B0):
            Q_i = Q[:, i:i+B0, :]
            O_i= torch.zeros((b, B0, d), device=Q.device, dtype=Q.dtype)
            l = torch.zeros((b, B0,), device=Q.device, dtype=Q.dtype)
            m = torch.full((b, B0,), float('-inf'), device=Q.device, dtype=Q.dtype)
            for j in range(0, s, B1):
                K_j, V_j = K[:, j:j+B1, :], V[:, j:j+B1, :] # (b, B1, d)
                S = einsum(Q_i, K_j, "b B_0 d_k, b B_1 d_k -> b B_0 B_1") * (d ** -0.5) # (b, B0, B1)
                m_blk = torch.maximum(m, torch.max(S, dim=-1).values) # (b, B0,)
                P = torch.exp(S - m_blk.unsqueeze(-1)) # (b, B0, B1)
                l = torch.exp(m - m_blk) * l + P.sum(dim=-1) # (b, B0,)
                O_i = torch.diag_embed(torch.exp(m - m_blk)) @ O_i + P @ V_j # (b, B0, d)
                m = m_blk
            O_i = torch.diag_embed(l ** -1.0) @ O_i # (b, B0, d)
            L_i = m + torch.log(l) # (b, B0,)
            O[:, i:i+B0, :] = O_i
            L[:, i:i+B0] = L_i
        ctx.save_for_backward(L, Q, K, V, O)
        return O
                
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

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
    # program id
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
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0), 
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q, K) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            mask = Q[:, None] + Q_TILE_SIZE * query_tile_index >= V[None, :] + K_TILE_SIZE * i
            S = tl.where(mask, S, -float('inf'))
        m_blk = tl.maximum(m, tl.max(S, axis=-1)) # (Q_TILE_SIZE,)
        P = tl.exp(S - m_blk[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        l = tl.exp(m - m_blk) * l + tl.sum(P, axis=-1) # (Q_TILE_SIZE,)
        O = tl.exp(m - m_blk)[:, None] * O + tl.dot(P, V) # (Q_TILE_SIZE, D)
        m = m_blk

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O = 1.0 / l[:, None] * O
    L = m + tl.log(l) # (Q_TILE_SIZE,)
    tl.store(O_block_ptr, O, boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))
        
        
    
class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        b, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        scale = D ** -0.5
        ctx.Q_TILE_SIZE = triton.next_power_of_2(N_QUERIES) // 4
        ctx.K_TILE_SIZE = triton.next_power_of_2(N_KEYS) // 4
        O = torch.zeros((b, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((b, N_QUERIES), device=Q.device, dtype=Q.dtype)
        ctx.save_for_backward(L, Q, K, V, O)
        flash_fwd_kernel[triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), b](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
        )  
        return O
        
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
        
def flashattn_spec(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) * (d ** -0.5) # (b, s, s) 
    P = torch.softmax(S, dim=-1) # (b, s, s)
    O = P @ V # (b, s, d)
    L = torch.logsumexp(S, dim=-1) # (b, s)
    return O

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
    Q = torch.randn((4, 256, 256), device=device, dtype=torch.float16)
    K = torch.randn((4, 256, 256), device=device, dtype=torch.float16)
    V = torch.randn((4, 256, 256), device=device, dtype=torch.float16)
    O_spec = flashattn_spec(Q, K, V)
    O = MyFlashAttnAutogradFunctionClass.apply(Q, K, V)
    print(f"Max absolute error: {(O - O_spec).abs().max()}")
    
