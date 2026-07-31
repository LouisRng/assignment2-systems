from __future__ import annotations
import torch
import os
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import math
from torch.optim import Optimizer
from torch import Tensor
from typing import Any, Type
from functools import wraps
from einops import einsum
from enum import Enum
from cs336_basics.model import Linear, Embedding

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500" 
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.handles = []
        self.module = module
        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param, 0, async_op=False)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self.hook)

    def hook(self, param):
        if param.requires_grad:
            self.handles.append(dist.all_reduce(param.grad, async_op=True))
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.div_(self.world_size)
        self.handles.clear()

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        
        self.world_size = dist.get_world_size()    
        self.rank = dist.get_rank()
        self._param_to_rank: dict[int, int] = {}
        self._local_param_groups: list[dict[str, Any]] = []
        self._next_param_index = 0
        self._constructed = False
        
        super().__init__(params, defaults=dict(kwargs))

        self._constructed = True

        # 如果当前 rank 没有分配到参数就不能创建优化器
        if len(self._local_param_groups) > 0:
            self.optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)
            self.state = self.optimizer.state 
        
    def step(self, closure=None, **kwargs):
        loss = None
        
        if self.optimizer is not None:
            loss = self.optimizer.step(closure=closure, **kwargs)
            
        for group in self.param_groups:
            for p in group["params"]:
                owner_rank = self._param_to_rank[id(p)]
                dist.broadcast(p.detach(), owner_rank, async_op=False)

        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' field")
            
        raw_param = param_group["params"]
        
        # 避免单个 Tensor 被当做 iterable 拆开
        if isinstance(raw_param, Tensor):
            params = [raw_param]
        else:
            params = list(raw_param) 

        full_param = dict(param_group)
        full_param["params"] = params
        
        # 完成参数组注册和本地参数筛选
        super().add_param_group(full_param)

        stored_group = self.param_groups[-1]
        local_params = []
        for p in stored_group["params"]:
            owner_rank = self._next_param_index % self.world_size
            self._param_to_rank[id(p)] = owner_rank
            if owner_rank == self.rank:
                local_params.append(p)
            self._next_param_index += 1

        if len(local_params) == 0:
            return 

        local_group = dict(stored_group)
        local_group["params"] = local_params
        
        # 动态添加参数组
        if not self._constructed:
            self._local_param_groups.append(local_group)
        else:
            # 某个 rank 在初始化期间有可能没有分到参数，optimizer 没有创建
            if self.optimizer is None:
                self.optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
                self.state = self.optimizer.state
            else:
                self.optimizer.add_param_group(local_group)
                self.state = self.optimizer.state
                
# 定义 GATHERING, READY, SHARDED
class Status(Enum):
    GATHERING = "gathering"
    READY = "ready"
    SHARDED = "sharded"
    IDLE = "idle"
    REDUCING = "reducing"
    SYNCED = "synced"

class FSDPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, wrapper: Wrapper):
        output = None
        ctx.wrapper = wrapper
        ctx.save_for_backward(x, shard)
        ctx.wrapper.get_full_weight()
        if isinstance(wrapper.module, Linear):
            output = einsum(x, ctx.wrapper.full_weight, '... d_in, d_out d_in -> ... d_out')
        if isinstance(wrapper.module, Embedding):
            output = ctx.wrapper.full_weight[x, :]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        ctx.wrapper.get_full_weight()
        
        grad_x = None
        grad_weight = None
        
        if isinstance(ctx.wrapper.module, Linear):
            grad_x = einsum(grad_output, ctx.wrapper.full_weight, "... d_out, d_out d_in -> ... d_in")
            grad_weight = einsum(grad_output, x, "... d_out , ... d_in -> d_out d_in")
        if isinstance(ctx.wrapper.module, Embedding):
            grad_weight = torch.zeros(ctx.wrapper.weight_shape, dtype=grad_output.dtype, device=grad_output.device)
            grad_weight.index_add_(0, x.reshape(-1), grad_output.reshape(-1, grad_output.shape[-1]))
        
        assert grad_weight.numel() == ctx.wrapper.weight_numel
            
        if not grad_weight.is_contiguous():
            raise ValueError
        grad_weight = grad_weight.view(-1)
        padding_right = ctx.wrapper.shard_size * ctx.wrapper.world_size - ctx.wrapper.weight_numel
        assert padding_right >= 0
        p1d = (0, padding_right)
        ctx.wrapper.reduce_scatter_input = F.pad(grad_weight, p1d, "constant", 0)
        ctx.wrapper.reduce_scatter_output = torch.empty(ctx.wrapper.shard_size, dtype=ctx.wrapper.reduce_scatter_input.dtype, device=ctx.wrapper.module.weight.device)
        
        if not ctx.wrapper.grad_state == Status.IDLE:
            raise ValueError
        ctx.wrapper.grad_buffer_handle = dist.reduce_scatter_tensor(ctx.wrapper.reduce_scatter_output, ctx.wrapper.reduce_scatter_input, async_op=True)
        ctx.wrapper.grad_state = Status.REDUCING

        ctx.wrapper.release_full_weight()
        ctx.wrapper.callback(ctx.wrapper)
        
        return grad_x, None, None
        
        

# 包装 submodule
class Wrapper(nn.Module):
    def __init__(self, module: nn.Module, world_size, rank, compute_dtype):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank
        self.compute_dtype = compute_dtype
        # module 元信息
        self.weight_shape = self.module.weight.shape
        self.weight_dtype = self.module.weight.dtype
        self.weight_device = self.module.weight.device
        self.weight_numel = self.module.weight.numel()
        
        self.lower_precision_weight = None
        self.shard_size = math.ceil(self.weight_numel / self.world_size)
        self.flatten_weight_buffer = None
        self.full_weight = None

        # 当前 rank shard, padding 信息
        start = self.shard_size * self.rank
        end = self.shard_size * (self.rank + 1)
        self.valid_length = max(0, min(end, self.weight_numel) - start)
        self.local_padding = self.shard_size - self.valid_length
        p1d = (0, self.local_padding)
        local_weight = self.module.weight.detach().reshape(-1)[start:end].clone()
        if self.local_padding > 0:
            local_weight = F.pad(local_weight, p1d, "constant", 0)
        # 注册当前 rank 持有的 weight shard
        self.module.weight = nn.Parameter(local_weight, requires_grad=self.module.weight.requires_grad)

        self.gather_weight_handle = None
        self.grad_buffer_handle = None
        
        self.weight_state = Status.SHARDED
        self.grad_state = Status.IDLE
        
        self.reduce_scatter_input = None
        self.reduce_scatter_output = None

    def forward(self, x):
        return FSDPFunction.apply(x, self.module.weight, self)
         
    # launch/prefetch mixed precision
    def start_weight_gather(self):
        if not self.weight_state == Status.SHARDED:
            raise ValueError
            
        with torch.no_grad():
            if self.compute_dtype is not None:
                self.lower_precision_weight = self.module.weight.to(self.compute_dtype) 
            else:
                self.lower_precision_weight = self.module.weight
                
            # 此时真正分配 buffer 空间
            if self.compute_dtype:
                self.flatten_weight_buffer = torch.empty(self.shard_size * self.world_size, dtype=self.compute_dtype, device=self.weight_device)
            else:
                self.flatten_weight_buffer = torch.empty(self.shard_size * self.world_size, dtype=self.weight_dtype, device=self.weight_device)
                
            self.gather_weight_handle = dist.all_gather_into_tensor(self.flatten_weight_buffer, self.lower_precision_weight, async_op=True)
            self.weight_state = Status.GATHERING
            
        

        return self.gather_weight_handle
    
    # consume/materialize
    def get_full_weight(self):
        if not self.weight_state == Status.GATHERING:
            raise ValueError
        
        self.gather_weight_handle.wait()
        self.full_weight = self.flatten_weight_buffer[:self.weight_numel].view(self.weight_shape)
        self.weight_state = Status.READY
        return self.full_weight

    def release_full_weight(self):
        if not self.weight_state == Status.READY:
            raise ValueError
        self.full_weight.untyped_storage().resize_(0)
        self.full_weight = None
        self.weight_state = Status.SHARDED
        self.gather_weight_handle = None
        self.lower_precision_weight = None
        

class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module 
        self.compute_dtype = compute_dtype

        # 调度 Wrapper
        self.wrappers_to_order = {}
        self.ordered_wrappers = []
        self.full_module = {}
        self.master_param_ids = set()
        self.replicated_params = []

        # hook handle
        self.all_reduce_hook_handles = []
        self.prefetch_weight_handles = []
        self.backward_bootstrap_handle = None
        self.replicated_grad_all_reduce_handles = []

        self.replicated_param_to_tensor = {}
        
        # 保证模型初始化参数一致
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, 0, async_op=False)

        order = 0 
        for name, submodule in self.module.named_modules():
            if isinstance(submodule, Linear) or isinstance(submodule, Embedding):
                wrapper = Wrapper(submodule, self.world_size, self.rank, self.compute_dtype)
                self.wrappers_to_order[wrapper] = order
                self.ordered_wrappers.append(wrapper)
                self.full_module[name] = wrapper
                order += 1

        for name, wrapper in self.full_module.items():
            self.master_param_ids.add(id(wrapper.module.weight))
            self.prefetch_weight_handles.append(wrapper.register_forward_hook(self.forward_weight_hook))
            wrapper.callback = self.callback
            self.module.set_submodule(name, wrapper)

        for name, param in self.module.named_parameters():
            if id(param) not in self.master_param_ids:
                if param.requires_grad:
                    self.replicated_params.append(param) 
                    self.all_reduce_hook_handles.append(param.register_post_accumulate_grad_hook(self.all_reduce_hook))
                    self.replicated_param_to_tensor[name] = param
            
        # 给 FSDP 自己注册 full backward pre hook 实现最后两层 backward 的 bootstrap
        self.backward_bootstrap_handle = self.register_full_backward_pre_hook(self.backward_weight_hook)

    def all_reduce_hook(self, param):
        if param.requires_grad:
            self.replicated_grad_all_reduce_handles.append(dist.all_reduce(param.grad, async_op=True))
                
        
    # forward post-hook
    def forward_weight_hook(self, module, args, output):
        idx = self.wrappers_to_order[module]
        # 释放当前层的 full weight
        module.release_full_weight()
        if idx + 2 < len(self.ordered_wrappers):
            wrapper = self.ordered_wrappers[idx + 2]
            # 开始 i+2 层的 weight gather
            if wrapper.weight_state == Status.SHARDED:
                wrapper.start_weight_gather()
        return None

    # backward pre-hook
    def backward_weight_hook(self, module, grad_output):
        idx = len(self.ordered_wrappers) - 1
        if self.ordered_wrappers[idx].weight_state != Status.SHARDED or self.ordered_wrappers[idx - 1].weight_state != Status.SHARDED:
            raise ValueError
        self.ordered_wrappers[idx].start_weight_gather()
        self.ordered_wrappers[idx - 1].start_weight_gather()
        return None
        
    # 第 i 层完成 backward 之后，启动 i - 2 层 weight gather
    def callback(self, wrapper: Wrapper):
        # 接收刚完成 backward 的 wrapper
        idx = self.wrappers_to_order[wrapper]
        if idx - 2 >= 0:
            wrapper = self.ordered_wrappers[idx - 2]
            if not wrapper.weight_state == Status.SHARDED:
                raise ValueError
            wrapper.start_weight_gather()
        return None
            
         
    def forward(self, *inputs, **kwargs):
        bootstrap = min(2, len(self.ordered_wrappers))
        for i in range(bootstrap):
            self.ordered_wrappers[i].start_weight_gather()
            
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for wrapper in self.ordered_wrappers:
            if not wrapper.grad_state == Status.REDUCING:
                raise ValueError
            wrapper.grad_buffer_handle.wait()
            wrapper.reduce_scatter_output = wrapper.reduce_scatter_output.to(torch.float32)
            wrapper.reduce_scatter_output.div_(self.world_size)
            assert wrapper.reduce_scatter_output.dtype == wrapper.module.weight.dtype
            assert wrapper.reduce_scatter_output.shape == wrapper.module.weight.shape
            assert wrapper.reduce_scatter_output.device == wrapper.module.weight.device
            # 支持梯度累加
            if wrapper.module.weight.grad is None:
                wrapper.module.weight.grad = wrapper.reduce_scatter_output
            else:
                wrapper.module.weight.grad += wrapper.reduce_scatter_output
            wrapper.reduce_scatter_input = None
            wrapper.reduce_scatter_output = None
            wrapper.grad_buffer_handle = None
            wrapper.grad_state = Status.IDLE
            
        for handle in self.replicated_grad_all_reduce_handles :
            handle.wait()
            
        for param in self.replicated_params:
            if param.grad is not None:
                param.grad.div_(self.world_size)
            
        self.replicated_grad_all_reduce_handles .clear()
        
        
        
          
        

    
                    





        
