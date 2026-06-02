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
        ''' Minimal implementation
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.world_size
        '''
        # All-reduce once
        flattened_grads = torch.cat([param.grad.data.view(-1) for param in self.module.parameters() if param.grad is not None])
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
        flattened_grads /= self.world_size
        
        # Distribute the reduced gradients back to each parameter
        grad_idx = 0
        for param in self.module.parameters():
            if param.grad is not None:
                grad_size = param.grad.data.numel()
                param.grad.data = flattened_grads[grad_idx:grad_idx + grad_size].view(param.grad.data.shape)
                grad_idx += grad_size