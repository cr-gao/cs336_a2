import math

from cs336_basics.model import Embedding, Linear
import torch
import torch.distributed as dist

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.fsdp_layers = []
        self.module = module
        self.wrap_layers()
        self.register_forward_hook()
        
        sharded_ids = {id(layer.shard_param) for layer in self.fsdp_layers}
        self.replicated_params = [p for p in self.module.parameters() if id(p) not in sharded_ids]
        
    def forward(self, *inputs, **kwargs):
        for i in range(min(2, len(self.fsdp_layers))):
            self.prefetch(i)

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for layer in self.fsdp_layers:
            layer.finish_grad()
            
        for p in self.replicated_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= self.world_size
            
    def wrap_layers(self):
        targets = [(n, c) for n, c in self.module.named_modules()
                   if isinstance(c, (Linear, Embedding))]
        for name, child in targets:
            wrapped = FSDPLeafModule(child, compute_dtype=self.compute_dtype, orig_name=name)
            *parent_path, attr = name.split(".")
            parent = self.module
            for p in parent_path:
                parent = getattr(parent, p)
            setattr(parent, attr, wrapped)
            self.fsdp_layers.append(wrapped)
                
    def register_forward_hook(self):
        for idx, layer in enumerate(self.fsdp_layers):
            target_idx = idx + 2
            
            def make_hook(current_layer, idx):
                def fwd_hook(module, input, output):
                    self.prefetch(idx)
                return fwd_hook
            layer.register_forward_hook(make_hook(layer, target_idx))
    
    def prefetch(self, idx):
        if idx >= len(self.fsdp_layers):
            return
        layer = self.fsdp_layers[idx]
        layer.gather()

class FSDPLeafModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None, orig_name: str = ""):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype
        self.orig_name = orig_name
        
        # Shard the weight parameter
        weight = self.module.weight.data
        self.orig_dim0 = weight.size(0)
        self.padded_dim0 = math.ceil(self.orig_dim0 / self.world_size) * self.world_size
        self.shard_size = self.padded_dim0 // self.world_size
        if self.padded_dim0 != self.orig_dim0:
            pad = weight.new_zeros((self.padded_dim0 - self.orig_dim0, *weight.shape[1:]))
            weight = torch.cat([weight, pad], dim=0)
        start = self.rank * self.shard_size
        shard = weight[start:start + self.shard_size].contiguous()
        self.module.weight = torch.nn.Parameter(shard)
        self.shard_box = [self.module.weight]

        self.full_param = None
        self.full_weight_list = None
        self.grad_shard = None
        self.gather_handle = None
        self.backward_handle = None
        
    @property
    def shard_param(self) -> torch.nn.Parameter:
        return self.shard_box[0]
        
    def forward(self, *inputs, **kwargs):
        if self.full_weight_list is None:
            self.gather()
            
        self.wait_and_assemble()
        self.full_param.register_hook(self.backward_hook)

        return self.module(*inputs, **kwargs)

    def gather(self):
        if self.gather_handle is not None or self.full_param is not None:
            return
        send = self.shard_param.data
        if self.compute_dtype is not None:
            send = send.to(self.compute_dtype)
        
        full_weight_list = [torch.zeros_like(send) for _ in range(self.world_size)]
        self.gather_handle = dist.all_gather(full_weight_list, send, async_op=True)
        self.full_weight_list = full_weight_list
    
    def free_full_weights(self):
        self.full_weight_list = None
        self.full_param = None
        self.module.weight = self.shard_param
        
    def wait_and_assemble(self):
        if self.gather_handle is not None:
            self.gather_handle.wait()
            self.gather_handle = None
        full_weight = torch.cat(self.full_weight_list, dim=0)[:self.orig_dim0].detach().requires_grad_(True)
        self.full_param = torch.nn.Parameter(full_weight)
        self.module.weight = self.full_param
        self.full_weight_list = None
        
    def backward_hook(self, grad):
        if self.padded_dim0 != self.orig_dim0:
            pad = grad.new_zeros((self.padded_dim0 - self.orig_dim0, *grad.shape[1:]))
            grad = torch.cat([grad, pad], dim=0)
        grad_shards = list(grad.chunk(self.world_size, dim=0))
        grad_output = torch.zeros_like(grad_shards[self.rank])
        self.backward_handle = dist.reduce_scatter(grad_output, grad_shards, async_op=True)
        self.grad_shard = grad_output
        return torch.zeros_like(grad)
    
    def finish_grad(self):
        if self.backward_handle is not None:
            self.backward_handle.wait()
            self.backward_handle = None
            grad = self.grad_shard / self.world_size
            if grad.dtype != self.shard_param.dtype:
                grad = grad.to(self.shard_param.dtype)
            self.shard_param.grad = grad
            self.grad_shard = None
        self.free_full_weights()