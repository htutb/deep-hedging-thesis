from abc import abstractmethod
import torch
import numpy as np
from ..instruments.Instruments import Instrument


class Primary(Instrument):
    '''
    Abstract class for primary instruments
    '''

    @abstractmethod
    def simulate(self, P, T) -> torch.Tensor:
        # returns P x T tensor of primary paths
        pass

    def value(self, primary_path):
        return primary_path

    def primary(self) -> Instrument:
        return self

    def delta(self, primary_path):
        return torch.ones_like(primary_path)


class GeometricBrownianStock(Primary):
    '''
    Geometric Brownian Motion stock price model
    Params:
        S0 (float) - initial stock price
        mu (float) - drift
        sigma (float) - volatility
    Input:
        P (int) - number of paths to simulate
        T (int) - number of timesteps to simulate
    Output:
        P x T tensor of primary paths
    '''

    def __init__(self, S0, mu, sigma):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def name(self):
        return f"Geometric Brownian Stock with S0={self.S0}, mu={self.mu}, sigma={self.sigma}"

    def simulate(self, P, T):
        return torch.cat([
            self.S0 * torch.ones(P, 1),
            self.S0 * torch.exp(torch.cumsum((self.mu - 0.5 * self.sigma ** 2) * torch.ones(P, T - 1) + self.sigma * torch.randn(P, T - 1), dim=1))
        ], dim=1)

# only returns stock price, not variance
class HestonStock(Primary):
    '''
    Heston stochastic volatility stock price model
    Params:
        S0 (float) - initial stock price
        V0 (float) - initial variance
        mu (float) - drift
        kappa (float) - mean reversion rate
        theta (float) - long-term variance
        xi (float) - volatility of volatility
        rho (float) - correlation between stock and variance
    Input:
        P (int) - number of paths to simulate
        T (int) - number of timesteps to simulate
    Output:
        P x T tensor of primary paths
    '''
    def __init__(self, S0, V0, mu, kappa, theta, xi, rho):
        self.S0 = S0
        self.V0 = V0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

        assert 2 * kappa * theta / xi**2 >= 1, f" {2 * kappa * theta / xi**2} is less that 1, Feller condition not satisfied, may lead to negative variance paths"

    def name(self):
        return f"Heston Stock with S0={self.S0}, V0={self.V0}, mu={self.mu}, kappa={self.kappa}, theta={self.theta}, xi={self.xi}, rho={self.rho}"

    def simulate(self, P, T):

        S = torch.zeros(P, T)
        V = torch.zeros(P, T)
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        step_size = 1 / (T - 1)
        stock_increments = torch.randn(P, T - 1) * np.sqrt(step_size)
        variance_increments = torch.randn(P, T - 1) * np.sqrt(step_size)
        variance_increments = (
            self.rho * stock_increments
            + np.sqrt(1 - self.rho ** 2) * variance_increments
        )

        for t in range(1, T):
            V[:, t] = (
                V[:, t - 1]
                + self.kappa * (self.theta - torch.clamp(V[:, t - 1], min=0)) * step_size
                + self.xi * torch.sqrt(torch.clamp(V[:, t - 1], min=0)) * variance_increments[:, t - 1]
            )
            S[:, t] = S[:, t - 1] * torch.exp(
                (self.mu - 0.5 * torch.clamp(V[:, t - 1], min=0)) * step_size
                + torch.sqrt(torch.clamp(V[:, t - 1], min=0)) * stock_increments[:, t - 1]
            )
        return S
    

class JumpStock(Primary):
    '''
    Jump diffusion stock price model
    Params:
        S0 (float) - initial stock price
        mu (float) - drift
        sigma (float) - volatility
        lam (float) - jump intensity
        m (float) - mean jump size
        v (float) - jump size volatility
    Input:
        P (int) - number of paths to simulate
        T (int) - number of timesteps to simulate
    Output:
        P x T tensor of primary paths
    '''
    def __init__(self, S0, mu, sigma, lam, m, v):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.m = m
        self.v = v

    def name(self):
        return f"Jump Stock with S0={self.S0}, mu={self.mu}, sigma={self.sigma}, lam={self.lam}, m={self.m}, v={self.v}"

    def simulate(self, P, T):
        dt = 1.0 / (T - 1)
        S = torch.zeros(P, T)
        S[:, 0] = self.S0
        kappa = torch.exp(torch.tensor(self.m + 0.5 * self.v ** 2)) - 1

        for t in range(1, T):
            Z = torch.randn(P)
            N = torch.poisson(torch.full((P,), self.lam * dt))
            
            jump_log = self.m * N + self.v * torch.sqrt(N) * torch.randn(P)
            J = torch.exp(jump_log)

            diffusion = torch.exp(
                (self.mu - self.lam * kappa - 0.5 * self.sigma ** 2) * dt
                + self.sigma * (dt ** 0.5) * Z
            )

            S[:, t] = S[:, t - 1] * diffusion * J

        return S
