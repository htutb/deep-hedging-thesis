from collections import OrderedDict
from typing import List
import torch
from ..Costs import CostFunction
from ..agents.Agent import Agent
from ..instruments.Instruments import Instrument
from ..optimizers import AdamOptimizer, SGLDOptimizer, SGHMCOptimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

class SimpleAgent(Agent):
    '''
    Simple feedforward agent that takes as input the log prices and time to maturity and outputs the action
    '''
    def __init__(self,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                cost_function: CostFunction,
                hedging_instruments: List[Instrument],
                interest_rate = 0.0,
                lr=0.005,
                use_gpu=True,
                h_dim=16,
                ):

        self.N = len(hedging_instruments)
        network_input_dim = self.input_dim()

        super().__init__(criterion, optimizer, cost_function, hedging_instruments, interest_rate, lr, use_gpu)
        self.network = torch.nn.Sequential(
        OrderedDict([
            ('fc1', torch.nn.Linear(network_input_dim, h_dim)),
            ('act1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(h_dim, h_dim)),
            ('act2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(h_dim, self.N))
        ])
    ).to(self.device)

        # self.optimizer = AdamOptimizer(self.network.parameters(), lr=lr)
        # self.optimizer = SGLDOptimizer(self.network.parameters(), lr=lr)
        # self.optimizer = SGHMCOptimizer(self.network.parameters(), lr=lr)

        self.optimizer = optimizer(self.network.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=2500)


    def input_dim(self) -> int:
        return self.N + 1


    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        # only know the current price
        paths, _, _, T = state
        P, t, N = paths.shape

        last_prices = paths[:, -1, :] # (P, N)
        # log prices
        log_prices = torch.log(last_prices) # (P, N)

        times = torch.ones(P, 1, device=self.device) * (T-t) # (P, 1)
        # features is log_prices and t
        features = torch.cat([log_prices, times], dim=1) # (P, N+1)

        return features.to(self.device)

    def forward(self, state: tuple) -> torch.Tensor:
        features = self.feature_transform(state) # D x input_dim
        return self.network(features)
