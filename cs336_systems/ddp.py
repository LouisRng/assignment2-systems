import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.optim as optim
from typing import Type, Any
import time
from collections.abc import Mapping
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW

DATA_PATH = Path(__file__).resolve().parent.parent / "result"
MERGE = DATA_PATH / "tinystories_bpe_merge.json"
VOCAB = DATA_PATH / "tinystories_bpe_vocab.json"
TRAIN_DATA = DATA_PATH / "TinyStoriesV2-GPT4-train_tokens.bin"
VAL_DATA = DATA_PATH / "TinyStoriesV2-GPT4-valid_tokens.bin"

# device = "cpu"
device = 'cuda' if torch.cuda.is_available() else \
        'mps' if torch.backends.mps.is_available() else 'cpu'
batch_size = 32
vocab_size = 10_000
context_length = 16
d_model = 2560
num_layers = 32
num_heads = 32
d_ff = 10240
rope_theta = 10000.0
betas = [0.9, 0.99]
weight_decay = 0.01
steps = 5
warmup = 0
lr = 1e-3 
eps = 1e-8

def load_data(path: str | os.PathLike) -> npt.NDArray:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return data

def get_batch(data, batch_size, context_length, device):
    ix = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[i: i+context_length] for i in ix])
    y = np.stack([data[i+1: i+context_length+1] for i in ix])
    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))
    return x.to(device), y.to(device)

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))

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

# benchmark naive ddp
def benchmark(rank, world_size, data, warmup):
    setup(rank, world_size)
    n = batch_size // world_size
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    # 验证参数是不是同步了
    # gathered = [None] * world_size
    # fingerprint = sum([param.sum() for param in ddp_model.parameters()])
    # dist.all_gather_object(gathered, fingerprint)
    # print(gathered)
    with torch.cuda.nvtx.range("broadcast parameters"):
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, 0, async_op=False)
    
    sliced_inputs = torch.empty((n, context_length), dtype=torch.int64, device=device)
    sliced_targets = torch.empty((n, context_length), dtype=torch.int64, device=device)
    # warmup
    elapsed_list = []
    elapsed_commu_grad_list = []
        
    for step in range(steps):
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.cuda.nvtx.range("warmup" if step < warmup else "benchmark"):
            if rank == 0:
                inputs, targets = get_batch(data, batch_size, context_length, device)
                inputs_scatter_list = [inputs[i * n: i * n + n, :] for i in range(world_size)]
                targets_scatter_list = [targets[i * n: i * n + n, :] for i in range(world_size)]
            else:
                inputs_scatter_list = None
                targets_scatter_list = None
            dist.scatter(sliced_inputs, inputs_scatter_list, src=0)
            dist.scatter(sliced_targets, targets_scatter_list, src=0)
            if step >= warmup:
                start = time.perf_counter()
            with torch.cuda.nvtx.range("forward"):
                logits = model(sliced_inputs)
            loss = cross_entropy(logits, sliced_targets)
            optimizer.zero_grad()
            
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
                
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= warmup:
                start_commu_grad = time.perf_counter()
            
            with torch.cuda.nvtx.range("all_reduce"):
                for param in model.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.grad, async_op=False)
                        param.grad.div_(world_size) 
                    
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= warmup:
                elapsed_commu_grad = time.perf_counter() - start_commu_grad
                elapsed_commu_grad_list.append(elapsed_commu_grad)
            with torch.cuda.nvtx.range("optimizer step"):
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= warmup:
                elapsed = time.perf_counter() - start
                elapsed_list.append(elapsed)
        
    elapsed = np.mean(elapsed_list)
    elapsed_commu_grad = np.mean(elapsed_commu_grad_list)
    print(f"rank:{rank} training time:{elapsed:.4f}s")
    print(f"rank:{rank} the proportion of time spent on gradients communication: {elapsed_commu_grad / elapsed * 100:.4f} %")

def benchmark_overlap_ddp(rank, world_size, data, warmup):
    setup(rank, world_size)
    n = batch_size // world_size
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    model.to(device)
    with torch.cuda.nvtx.range("broadcast parameters"):
        ddp_model = DDP(model)
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    sliced_inputs = torch.empty((n, context_length), dtype=torch.int64, device=device)
    sliced_targets = torch.empty((n, context_length), dtype=torch.int64, device=device)
    # warmup
    elapsed_list = []
    
    for step in range(steps):
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.cuda.nvtx.range("warmup" if step < warmup else "benchmark"):
            if rank == 0:
                inputs, targets = get_batch(data, batch_size, context_length, device)
                inputs_scatter_list = [inputs[i * n: i * n + n, :] for i in range(world_size)]
                targets_scatter_list = [targets[i * n: i * n + n, :] for i in range(world_size)]
            else:
                inputs_scatter_list = None
                targets_scatter_list = None
            dist.scatter(sliced_inputs, inputs_scatter_list, src=0)
            dist.scatter(sliced_targets, targets_scatter_list, src=0)
            if step >= warmup:
                start = time.perf_counter()
            with torch.cuda.nvtx.range("forward"):
                logits = ddp_model(sliced_inputs)
            loss = cross_entropy(logits, sliced_targets)
            optimizer.zero_grad()
            
            with torch.cuda.nvtx.range("backward"):
                loss.backward()

            # 优化器依赖于梯度，因此在这里调用finish_gradient_synchronization()，以确保梯度通信完成后再进行优化器步骤
            with torch.cuda.nvtx.range("finish_gradient_synchronization"):
                ddp_model.finish_gradient_synchronization()
            with torch.cuda.nvtx.range("optimizer step"):
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= warmup:
                elapsed = time.perf_counter() - start
                elapsed_list.append(elapsed)

    elapsed = np.mean(elapsed_list)
    print(f"rank:{rank} training time:{elapsed:.4f}s")

class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_param_groups = []
        self.param_to_rank = {}
        self.all_params = []
        self.params = list(params)
        self.global_idx = 0
        self.wrapped_optimizer = None
            
        super().__init__(self.params, kwargs)
                
        if len(self.local_param_groups) != 0:
            self.wrapped_optimizer = optimizer_cls(self.local_param_groups, **kwargs)
        else:
            self.wrapped_optimizer = None
                
    def step(self, closure=None, **kwargs):
        if self.wrapped_optimizer is not None:
            loss = self.wrapped_optimizer.step(closure, **kwargs)
        else:
            loss = None 
        with torch.no_grad():
            for param in self.all_params:
                rank = self.param_to_rank[param]
                dist.broadcast(param, rank, async_op=False)
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        # param_group["params"] 有可能为 generator，避免被父类消耗掉
        params_list = list(param_group["params"])
        full_group = dict(param_group)
        full_group["params"] = new_group
        optim.Optimizer.add_param_group(self, full_group)
        
        local_group = dict(full_group)
        local_group["params"] = []
        for param in params_list:
            self.all_params.append(param) 
            self.param_to_rank[param] = self.global_idx % self.world_size
            if self.global_idx % self.world_size == self.rank:
                local_group["params"].append(param)
            self.global_idx += 1
        if len(local_group["params"]) != 0:
            self.local_param_groups.append(local_group)
            if self.wrapped_optimizer is not None:
                self.wrapped_optimizer.add_param_group(local_group)

def benchmark_optimizer(rank, world_size, data, sharded_optimizer=False):
    setup(rank, world_size)
    n = batch_size // world_size
    torch.cuda.reset_peak_memory_stats()
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    model.to(device)
    ddp_model = DDP(model)
    if sharded_optimizer:
        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        torch.cuda.synchronize()
        print(f"benchmark memory peak usage with optimizer state sharding: training model with world_size={world_size}, batch_size={batch_size}, context_length={context_length}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}, rope_theta={rope_theta} on {device} using {world_size} processes, optimizer: {optimizer.__class__.__name__}")
        print(f"rank:{rank} after sharded optimizer initialization, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"rank:{rank} after sharded optimizer initialization, current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"rank:{rank} the optimizer state of AdamW is initialized lazily, so no exp_avg and exp_avg_sq are allocated yet, only the parameters are allocated")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        torch.cuda.synchronize()
        print(f"benchmark memory peak usage without optimizer state sharding: training model with world_size={world_size}, batch_size={batch_size}, context_length={context_length}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}, rope_theta={rope_theta} on {device} using {world_size} processes, optimizer: {optimizer.__class__.__name__}")
        print(f"rank:{rank} after AdamW initialization, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"rank:{rank} after AdamW initialization, current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    
    sliced_x = torch.empty((n, context_length), dtype=torch.int64, device=device)
    sliced_y = torch.empty((n, context_length), dtype=torch.int64, device=device)
    for step in range(steps):
        optimizer.zero_grad()
        if rank == 0:
            x, y = get_batch(data, batch_size, context_length, device)
            scattered_x_list = [x[i * n: i * n + n, :] for i in range(world_size)]
            scattered_y_list = [y[i * n: i * n + n, :] for i in range(world_size)]
        else:
            scattered_x_list = None
            scattered_y_list = None
        dist.scatter(sliced_x, scattered_x_list, src=0)
        dist.scatter(sliced_y, scattered_y_list, src=0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if step == 0:
            print(f"rank:{rank} model parameters memory usage: {sum([param.numel() * param.element_size() for param in ddp_model.parameters()]) / 1024 ** 2:.2f} MB")
        logits = ddp_model(sliced_x) 
        loss = cross_entropy(logits, sliced_y) 
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        if step == 0:
            print(f"rank:{rank} gradients memory usage: {sum([param.grad.numel() * param.grad.element_size() for param in ddp_model.parameters() if param.grad is not None]) / 1024 ** 2:.2f} MB")
            print(f"rank:{rank} after first backward and before optimizer step, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"rank:{rank} after first backward and before optimizer step, current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        torch.cuda.synchronize()
        if step == 0:
            if sharded_optimizer:
                print(f"rank:{rank} sharded optimizer state memory usage: {2 * sum([param.numel() * param.element_size() \
                for param in optimizer.all_params if optimizer.param_to_rank[param] == rank]) / 1024 ** 2:.2f} MB")
            else:
                total = 0
                for state_for_one_param in optimizer.state.values():
                    for value in state_for_one_param.values():
                        if isinstance(value, torch.Tensor):
                            total += value.numel() * value.element_size()
                print(f"rank:{rank} non-sharded optimizer state memory usage: {total / 1024 ** 2:.2f} MB")
            print(f"rank:{rank} after optimizer step, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"rank:{rank} after optimizer step, current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        
    
if __name__ == "__main__":
    world_size = 2
    data = load_data(TRAIN_DATA)
    sharded_optimizer = True
    mp.spawn(fn=benchmark_optimizer, args=(world_size, data, sharded_optimizer), nprocs=world_size, join=True)
    print(f"-"*50)
    mp.spawn(fn=benchmark_optimizer, args=(world_size, data, not sharded_optimizer), nprocs=world_size, join=True)
    
    
    # print(f"naive ddp: training model with world_size={world_size}, batch_size={batch_size}, context_length={context_length}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}, rope_theta={rope_theta} on {device} using {world_size} processes")
    # mp.spawn(fn=benchmark, args=(world_size, data, warmup), nprocs=world_size, join=True) 
    # print(f"overlap ddp: training model with world_size={world_size}, batch_size={batch_size}, context_length={context_length}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}, rope_theta={rope_theta} on {device} using {world_size} processes")
    # mp.spawn(fn=benchmark_overlap_ddp, args=(world_size, data, warmup), nprocs=world_size, join=True) 

    
