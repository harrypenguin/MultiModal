import numpy as np
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.warmup:
            return epoch / float(self.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup)))
