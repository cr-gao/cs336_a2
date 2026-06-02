import torch
import torch.distributed as dist
import time

class DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        total_numel = 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            total_numel += param.numel()
            
        self.flattened_grads = torch.zeros(total_numel, device=next(module.parameters()).device)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def sync_gradients(self, optimizer):
        ''' Minimal implementation
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size
        '''
        grad_idx = 0
        for param in self.module.parameters():
            if param.grad is not None:
                grad_size = param.grad.numel()
                self.flattened_grads[grad_idx:grad_idx + grad_size].copy_(param.grad.view(-1))
                grad_idx += grad_size

        # All-reduce once
        dist.all_reduce(self.flattened_grads, op=dist.ReduceOp.SUM)
        self.flattened_grads /= self.world_size

        # Distribute the reduced gradients back to each parameter
        grad_idx = 0
        for param in self.module.parameters():
            if param.grad is not None:
                grad_size = param.grad.numel()
                param.grad.copy_(self.flattened_grads[grad_idx:grad_idx + grad_size].view_as(param.grad))
                grad_idx += grad_size