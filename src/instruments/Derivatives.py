import torch
from torch.distributions import Normal
from ..instruments.Claims import Claim
from ..instruments.Instruments import Instrument
from ..instruments.Primaries import Primary


class Derivative(Claim):
    '''
    Abstract class for derivatives (claims that depend on primaries)
    '''
    def __init__(self, primary: Primary):
        self._primary = primary

    def primary(self) -> Primary:
        return self._primary

    def delta(self, primary_path):
        pass


class EuropeanOption(Derivative):
    '''
    Abstract class for European options
    Params:
        primary (Primary) - primary underlying the option
        strike (float) - strike price
        expiry (int) - expiry time in timesteps
    '''
    def __init__(self, primary: Primary, strike: float, expiry: float):
        super().__init__(primary)
        self.strike = strike
        self.expiry = expiry


class EuropeanCall(EuropeanOption):
    '''
    European call option
        payoff = max(S_T - K, 0)
    Inputs:
        primary_path (P x T tensor) - paths of the primary underlying the option
    Output:        
        payoff (P tensor) - payoff of the option at expiry
    '''
    def payoff(self, primary_path):
        return torch.clamp(primary_path[:,-1] - self.strike, min=0)

class EuropeanPut(EuropeanOption):
    '''
    European put option
        payoff = max(K - S_T, 0)
    Inputs:
        primary_path (P x T tensor) - paths of the primary underlying the option
    Output:        
        payoff (P tensor) - payoff of the option at expiry
    '''
    def payoff(self, primary_path):
        return torch.clamp(self.strike - primary_path[:,-1], min=0)

class BSCall(EuropeanCall, Instrument):
    '''
    Black-Scholes European call option
    Params:
        primary (Primary) - primary underlying the option
        strike (float) - strike price
        expiry (float) - expiry time in timesteps
        drift (float) - drift of the primary
        volatility (float) - volatility of the primary
    Inputs:
        primary_path (P x T tensor) - paths of the primary underlying the option
    Output:        
        value (P x T tensor) - value of the option at each timestep
        delta (P x T tensor) - delta of the option at each timestep
    '''

    def __init__(self, primary: Primary, strike: float, expiry: float, drift: float, volatility: float):
        super().__init__(primary, strike, expiry)
        self.drift = drift
        self.volatility = volatility
        self.norm = Normal(0, 1)

    def value(self, primary_path) -> torch.Tensor:
        expiries = self.expiry - (torch.arange(primary_path.shape[1], device=primary_path.device) + 1) * (self.expiry / primary_path.shape[1])
        expiries = expiries.unsqueeze(0)
        eps = 1e-8
        sqrt_exp = torch.sqrt(expiries + eps)

        d1 = (torch.log(primary_path/self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (self.volatility * sqrt_exp)
        d2 = d1 - self.volatility * sqrt_exp

        value = primary_path * self.norm.cdf(d1) - self.strike * torch.exp(-self.drift * expiries) * self.norm.cdf(d2)
        value[:,-1] = torch.clamp(primary_path[:,-1] - self.strike, min=0)
        return value

    def delta(self, primary_path) -> torch.Tensor:
        expiries = self.expiry - (torch.arange(primary_path.shape[1], device=primary_path.device) + 1) * (self.expiry / primary_path.shape[1])
        expiries = expiries.unsqueeze(0)
        eps = 1e-8
        sqrt_exp = torch.sqrt(expiries + eps)
        
        d1 = (torch.log(primary_path/self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (self.volatility * sqrt_exp)
        return self.norm.cdf(d1)


class BSPut(EuropeanPut, Instrument):
        '''
        Black-Scholes European put option
        Params:
            primary (Primary) - primary underlying the option
            strike (float) - strike price
            expiry (float) - expiry time in timesteps
            drift (float) - drift of the primary
            volatility (float) - volatility of the primary
        Inputs:
            primary_path (P x T tensor) - paths of the primary underlying the option
        Outputs:
            value (P x T tensor) - value of the option at each timestep
            delta (P x T tensor) - delta of the option at each timestep
        '''
        def __init__(self, primary: Primary, strike: float, expiry: float, drift: float, volatility: float):
            super().__init__(primary, strike, expiry)
            self.drift = drift
            self.volatility = volatility
            self.norm = Normal(0, 1)

        def value(self, primary_path) -> torch.Tensor:
            expiries = self.expiry - (torch.arange(primary_path.shape[1], device=primary_path.device) + 1) * (self.expiry / primary_path.shape[1])
            expiries = expiries.unsqueeze(0)
            eps = 1e-8
            sqrt_exp = torch.sqrt(expiries + eps)
            
            d1 = (torch.log(primary_path/self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (self.volatility * sqrt_exp)
            d2 = d1 - self.volatility * sqrt_exp

            value = self.strike * torch.exp(-self.drift * expiries) * self.norm.cdf(-d2) - primary_path * self.norm.cdf(-d1)
            value[:,-1] = torch.clamp(self.strike - primary_path[:,-1], min=0)
            return value

        def delta(self, primary_path) -> torch.Tensor:
            expiries = self.expiry - (torch.arange(primary_path.shape[1], device=primary_path.device) + 1) * (self.expiry / primary_path.shape[1])
            expiries = expiries.unsqueeze(0)
            eps = 1e-8
            sqrt_exp = torch.sqrt(expiries + eps)
            
            d1 = (torch.log(primary_path/self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (self.volatility * sqrt_exp)
            return self.norm.cdf(d1) - 1
