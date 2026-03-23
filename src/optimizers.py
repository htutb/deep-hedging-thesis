import torch
from torch.optim import Adam


class AdamOptimizer(Adam):
    '''
    Adam optimizer with default parameters
    '''
    def __init__(self, params, lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super(AdamOptimizer, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def name(self):
        return "Adam"

    def step(self):
        super(AdamOptimizer, self).step()
    

class SGLDOptimizer(torch.optim.Optimizer):
    '''
    Stochastic Gradient Langevin Dynamics optimizer
    '''
    def __init__(self, params, lr=0.005, weight_decay=0.0, temp=0.000001):
        defaults = dict(lr=lr, weight_decay=weight_decay, temp=temp)
        super().__init__(params, defaults)

    def name(self):
        return "SGLD"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            temp = group["temp"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.clone() 

                if weight_decay != 0.0: 
                    grad.add_(p, alpha=weight_decay)

                noise = torch.randn_like(p) * (lr * temp) ** 0.5

                p.add_(grad, alpha=-lr)
                p.add_(noise)

        return loss


class SGHMCOptimizer(torch.optim.Optimizer):
    '''
    Stochastic Gradient Hamiltonian Monte Carlo
    '''
    def __init__(self, params, lr=0.005, weight_decay=0.0, momentum=0.9, temperature=0.00005):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, temperature=temperature)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['velocity'] = torch.zeros_like(p.data)

    def name(self):
        return "SGHMC"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            temperature = group['temperature']
            alpha = 1.0 - momentum  # friction

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clone()
                if weight_decay != 0.0:
                    grad.add_(p, alpha=weight_decay)

                v = self.state[p]['velocity']

                # N(0, 2 * alpha * lr * T)
                noise = torch.randn_like(p) * (2.0 * alpha * lr * temperature) ** 0.5
                v.mul_(momentum).add_(grad, alpha=-lr).add_(noise)
                p.add_(v)

        return loss