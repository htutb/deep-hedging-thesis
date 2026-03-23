from abc import ABC, abstractmethod
import torch


class CostFunction(ABC):
    '''
    Abstract base class for cost functions.
    '''
    @abstractmethod
    def cost(self, action, state = None) -> torch.Tensor:
        pass

    def __call__(self, action, state = None):
        return self.cost(action, state)

    def __add__(self, other):
        return SumCost(self, other)


class SumCost(CostFunction):
    '''
    Cost function that sums multiple cost functions.
    '''
    def __init__(self, *costs):
        self.costs = costs

    def cost(self, action, state):
        return sum(c.cost(action, state) for c in self.costs)


class FixedCost(CostFunction):
    '''
    Returns the fixed cost of the action.
    Params:
        fixed_cost (float) - fixed cost (e.g. 0.01 means 1% cost)
    Input:
        action: B x D (action taken at time t, where B is batch size and D is the number of assets)
        state: tuple of tensors (hedge_paths[:,:t], cash_account[:,:t], positions[:,:t], portfolio_value[:,:t])
    Output:
        (B, ) (fixed cost of the action)
    '''

    def __init__(self, fixed_cost):
        self.fixed_cost = fixed_cost

    def cost(self, action, state):
        return self.fixed_cost * (action != 0).sum(dim=-1)


class ProportionalCost(CostFunction):
    '''
    Returns the proportional cost of the action.
    Params:
        cost_rate (float) - proportional cost rate (e.g. 0.01 means 1% cost)
    Input:
        action: B x D (action taken at time t, where B is batch size and D is the number of assets)
        state: tuple of tensors (hedge_paths[:,:t], cash_account[:,:t], positions[:,:t], portfolio_value[:,:t])
    Output:
        (B, ) (proportional cost of the action)
    '''
    def __init__(self, cost_rate):
        self.cost_rate = cost_rate

    def cost(self, action, state):
        purchase_price = state[0][:, -1, :] * action
        return self.cost_rate * purchase_price.abs().sum(dim=-1)
