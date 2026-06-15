import torch
import triton
import triton.language as tl
from einops import einsum

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
    
    # on chip buffers 精度应该该是float32，避免数值不稳定
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q_offs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   # (Q_TILE_SIZE,)
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q, K) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            k_offs = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)              # (K_TILE_SIZE,)
            causal_mask = q_offs[:, None] >= k_offs[None, :]                  # (Q, K) tile
            S = tl.where(causal_mask, S, -1e6)
        m_blk = tl.maximum(m, tl.max(S, axis=-1)) # (Q_TILE_SIZE,)  
        P = tl.exp(S - m_blk[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        l = tl.exp(m - m_blk) * l + tl.sum(P, axis=-1) # (Q_TILE_SIZE,)
        O = tl.exp(m - m_blk)[:, None] * O
        P_cast = P.to(V.dtype)
        O = tl.dot(P_cast, V, acc=O) # (Q_TILE_SIZE, D)
        m = m_blk

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O = 1.0 / l[:, None] * O
    L = m + tl.log(l) # (Q_TILE_SIZE,)

    # O 的精度和输入保持一致，L 的精度保持为float32，避免数值不稳定
    tl.store(O_block_ptr, O.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


@torch.compile
def flashattn_backward(L, Q, K, V, O, scale, grad_out, is_causal=False):
    print(f"Running backward with grad_out dtype: {grad_out.dtype}, Q dtype: {Q.dtype}, K dtype: {K.dtype}, V dtype: {V.dtype}, O dtype: {O.dtype}")
    grad_out = grad_out.float()
    D = torch.sum(O * grad_out, dim=-1) # (b, s)
    S = Q.float() @ K.float().transpose(-2, -1) * scale
    if is_causal:
        mask = torch.tril(torch.ones_like(S, dtype=bool))
        S = torch.where(mask, S, torch.fill(torch.empty_like(S), -1e6))
    P = torch.exp(S - L[:, :, None]) # (b, s, d) - (b, s)
    dV = P.transpose(-2, -1) @ grad_out
    dP = grad_out @ V.transpose(-2, -1)
    dS = P * (dP - D[:, :, None])
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV, None

class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        b, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        scale = D ** -0.5
        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE = triton.next_power_of_2(N_QUERIES) // 4
        ctx.K_TILE_SIZE = triton.next_power_of_2(N_KEYS) // 4
        O = torch.zeros((b, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        
        # L 用于存储每个查询的 logsumexp 的结果，精度保持为float32，避免数值不稳定
        L = torch.empty((b, N_QUERIES), device=Q.device, dtype=torch.float32)
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
        L, Q, K, V, O = ctx.saved_tensors
        scale = Q.shape[-1] ** -0.5
        return flashattn_backward(L, Q, K, V, O, scale, grad_out, ctx.is_causal)


class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    def __init__(self):
        super().__init__() 

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B0, B1 = 16, 16
        b, N_q, d_k = Q.shape
        _, N_k, d_v = V.shape
        scale = d_k ** -0.5
        L = torch.empty((b, N_q,), device=Q.device, dtype=Q.dtype)
        O = torch.empty((b, N_q, d_k), device=Q.device, dtype=Q.dtype)
        for i in range(0, N_q, B0):
            Q_i = Q[:, i:i+B0, :]
            O_i = torch.zeros((b, B0, d_k), device=Q.device, dtype=Q.dtype)
            l = torch.zeros((b, B0,), device=Q.device, dtype=Q.dtype)
            m = torch.full((b, B0,), float('-inf'), device=Q.device, dtype=Q.dtype)
            for j in range(0, N_k, B1):
                K_j, V_j = K[:, j:j+B1, :], V[:, j:j+B1, :] # (b, B1, d)
                S = einsum(Q_i, K_j, "b B_0 d_k, b B_1 d_k -> b B_0 B_1") * scale # (b, B0, B1)
                if is_causal:
                    q_offs = torch.arange(i, min(i+B0, N_q), device=Q.device)[None, :] # (1, B0)
                    k_offs = torch.arange(j, min(j+B1, N_k), device=Q.device)[:, None] # (B1, 1)
                    causal_mask = q_offs >= k_offs # (B0, B1)
                    S = torch.where(causal_mask[None, :, :], S, torch.tensor(-float("inf"), device=Q.device, dtype=Q.dtype))
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
        ctx.is_causal = is_causal
        return O
                
    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        scale = Q.shape[-1] ** -0.5
        return flashattn_backward(L, Q, K, V, O, scale, grad_out, ctx.is_causal)

def benchmark_pytorch_flash_attn():
    device = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
    for i in range(7, 17):
        for j in range(4, 8):
            print(f"Benchmarking Pytorch Flash Attention with Q/K/V shape: (1, {2 ** i}, {2 ** j})")
            Q = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            K = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            V = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            # out 的精度和输入是一致的
            out = MyTritonFlashAttentionAutogradFunctionClass.apply(Q, K, V, True)
            out.sum().backward()
            
def benchmark_triton_flash_attn():
    device = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else 'cpu'
    for i in range(7, 17):
        for j in range(4, 8):
            print(f"Benchmarking Triton Flash Attention with Q/K/V shape: (1, {2 ** i}, {2 ** j})")
            Q = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            K = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            V = torch.randn((1, 2 ** i, 2 ** j), device=device, dtype=torch.bfloat16, requires_grad=True)
            out = MyFlashAttnAutogradFunctionClass.apply(Q, K, V, True)
            out.sum().backward()

if __name__ == "__main__":
    print("Benchmarking Pytorch Flash Attention Implementation")
    triton.testing.do_bench(benchmark_pytorch_flash_attn(), warmup=5, rep=25)

    print("-" * 50)

    print("Benchmarking Triton Flash Attention Implementation")
    triton.testing.do_bench(benchmark_triton_flash_attn(), warmup=5, rep=25)

    
    
    
    
