import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import time
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW

DATA_PATH = Path(__file__).resolve().parent.parent / "result"
MERGE = DATA_PATH / "tinystories_bpe_merge.json"
VOCAB = DATA_PATH / "tinystories_bpe_vocab.json"
TRAIN_DATA = DATA_PATH / "TinyStoriesV2-GPT4-train_tokens.bin"
VAL_DATA = DATA_PATH / "TinyStoriesV2-GPT4-valid_tokens.bin"

device = "cpu"
# device = 'cuda' if torch.cuda.is_available() else \
#         'mps' if torch.backends.mps.is_available() else 'cpu'
batch_size = 8
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
warmup = 5

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
        self.handles = []
        

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500" 
    # torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        # device_id=torch.device(f"cuda:{rank}"),
    )

def benchmark(rank, world_size, data, warmup):
    setup(rank, world_size)
    n = batch_size // world_size
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=betas, eps=1e-8, weight_decay=weight_decay)
    
    # 验证参数是不是同步了
    # gathered = [None] * world_size
    # fingerprint = sum([param.sum() for param in ddp_model.parameters()])
    # dist.all_gather_object(gathered, fingerprint)
    # print(gathered)
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, 0, async_op=False)
    
    sliced_inputs = torch.empty((n, context_length), dtype=torch.int64)
    sliced_targets = torch.empty((n, context_length), dtype=torch.int64)
    # warmup
    elapsed_list = []
    elapsed_commu_grad_list = []
        
    for step in range(steps):
        if device == "cuda":
            torch.cuda.synchronize()
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
        logits = model(sliced_inputs)
        loss = cross_entropy(logits, sliced_targets)
        optimizer.zero_grad()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        if step >= warmup:
            start_commu_grad = time.perf_counter()
        for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad, async_op=False)
                param.grad.div_(world_size) 
        if device == "cuda":
            torch.cuda.synchronize()
        if step >= warmup:
            elapsed_commu_grad = time.perf_counter() - start_commu_grad
            elapsed_commu_grad_list.append(elapsed_commu_grad)
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
     

    
if __name__ == "__main__":
    world_size = 4
    data = load_data(TRAIN_DATA)
    mp.spawn(fn=benchmark, args=(world_size, data, warmup), nprocs=world_size, join=True) 
        



