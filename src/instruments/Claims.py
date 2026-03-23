from abc import ABC, abstractmethod
import torch

# claim -> instrument -> [derivatives, primary]

class Claim(ABC):
    '''
    Abstract class for claims (payoff functions of derivatives).
    '''

    @abstractmethod
    def payoff(self, primary_path) -> torch.Tensor:
        pass

    @abstractmethod
    def primary(self):
        # return primary of the claim
        pass

    def delta(self, primary_path):
        return None
