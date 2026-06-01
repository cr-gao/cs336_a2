import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def sync_gradients(self, optimizer):
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size