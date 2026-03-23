from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskMeasure(ABC, torch.nn.Module):
    '''
    Abstract class for risk measures.
    '''
    pass


class WorstCase(RiskMeasure):
    '''
    Returns the worst portfolio value among all paths.
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (worst portfolio value)
    '''
    def forward(self, portfolio_value: torch.Tensor):
        return portfolio_value.min()


class TailValue(RiskMeasure):
    '''
    Returns the mean of the worst alpha % of the paths
    Params:
        alpha (float) - what % of worst paths to take
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (mean of the worst alpha % of the paths)
    '''
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        k = int(self.alpha * portfolio_value.shape[0])
        return portfolio_value.topk(k, largest=False).values.mean()


class Median(RiskMeasure):
    '''
    Returns the median of all portfolio values
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (median of portfolio values)
    '''
    def forward(self, portfolio_value: torch.Tensor):
        return portfolio_value.median()


class Expectation(RiskMeasure):
    '''
    Returns the mean of all portfolio values
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (mean of portfolio values)
    '''
    def forward(self, portfolio_value: torch.Tensor):
        return portfolio_value.mean()


class Entropy(RiskMeasure):
    '''
    Returns the entropy of the portfolio values
    Params:
        lambd (float) - risk aversion parameter (higher means more risk averse)
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (entropy of portfolio values)'''
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, portfolio_value: torch.Tensor):
        return - 1/self.lambd * torch.log(torch.exp(-self.lambd*portfolio_value).mean())


class CVaR(torch.nn.Module):
    '''
    Returns the Conditional Value at Risk (CVaR) of the portfolio values
    Params:
        p (float) - confidence level (0.95 means the worst 5% of the paths)
        beta (float) - smoothing parameter for the softplus function (higher means smoother approximation)
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (CVaR of portfolio values)
    '''
    def __init__(self, p: float, beta: float = 1e-2, omega: float = 0.0):
        super().__init__()
        self.p = p
        self.beta = beta
        self.omega = nn.Parameter(torch.tensor(float(omega), dtype=torch.float32))

    def forward(self, portfolio_value: torch.Tensor):
        x = portfolio_value.view(-1)
        loss = -x

        # VaR threshold
        omega = self.omega.to(device=x.device, dtype=x.dtype)

        # smooth version of max(0, z)
        excess = F.softplus((loss - omega) / self.beta) * self.beta
        cvar_loss = omega + (1.0 / (1 - self.p)) * excess.mean()

        return -cvar_loss


class Variance(RiskMeasure):
    '''
    Returns the negative variance of the portfolio values
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (negative variance of portfolio values)
    '''
    def forward(self, portfolio_value: torch.Tensor):
        return - portfolio_value.var()


class ExpectationVariance(RiskMeasure):
    '''
    Returns the mean of the portfolio values minus alpha times the variance of the portfolio values
    Params:
        alpha (float) - risk aversion parameter (higher means more risk averse)
    Input:
        portfolio_value: P x 1 (final portfolio value for every path)
    Output:
        1 x 1 (negative variance of portfolio values)
    '''
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):

        return portfolio_value.mean() - self.alpha * portfolio_value.var()
