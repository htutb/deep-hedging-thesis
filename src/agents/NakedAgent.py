from typing import List
import torch
from ..Costs import CostFunction
from ..agents.Agent import Agent
from ..instruments.Instruments import Instrument


class NakedAgent(Agent):
    '''
    Agent that does not hedge at all, i.e. always returns zero action
    '''
    def __init__(self,
                criterion: torch.nn.Module,
                cost_function: CostFunction,
                hedging_instruments: List[Instrument],
                stock_params,
                interest_rate = 0.0,
                lr=0.005,
                use_gpu=True,
                h_dim=15,
                ):
        super().__init__(criterion, cost_function, hedging_instruments, interest_rate, lr, use_gpu)

    def forward(self, state: tuple) -> torch.Tensor:
        return torch.zeros(self.N, device=self.device)
