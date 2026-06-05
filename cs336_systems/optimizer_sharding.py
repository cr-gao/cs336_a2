import torch
import torch.distributed as dist

from torch.optim import Optimizer
from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.initializing = True
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.all_params = list(params)
        self.param_to_rank = {}
        local_params = self.shard_params()
        
        super().__init__(local_params, {})
        self.optimizer = optimizer_cls(local_params, **kwargs)
        self.initializing = False
        
    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)
        self.sync_parameters()
        
    def add_param_group(self, param_group: dict[str, Any]):
        if self.initializing:
            super().add_param_group(param_group)
            return
        local_params = []
        for param in param_group['params']:
            global_idx = len(self.all_params)
            self.all_params.append(param)
            
            owner = global_idx % self.world_size
            self.param_to_rank[global_idx] = owner
            if owner == self.rank:
                local_params.append(param)
        
        if local_params:
            super().add_param_group({'params': local_params})
            self.optimizer.add_param_group({'params': local_params})
    
    def shard_params(self):
        local_params = []
        for idx, param in enumerate(self.all_params):
            owner = idx % self.world_size
            self.param_to_rank[idx] = owner
            if owner == self.rank:
                local_params.append(param)
        return local_params

    def sync_parameters(self):
        with torch.no_grad():
            dist.barrier()
            for idx, param in enumerate(self.all_params):
                dist.broadcast(param.data, src=self.param_to_rank[idx])