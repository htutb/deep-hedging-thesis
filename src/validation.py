from typing import List
import torch

from src.Costs import ProportionalCost

from src.instruments.Claims import Claim
from src.instruments.Derivatives import BSCall
from src.instruments.Instruments import Instrument
from src.instruments.Primaries import HestonStock

import src.RiskMeasures as RiskMeasures

from src.ExperimentRunner import validate_from_weights


def validation():
    seed = 133
    torch.manual_seed(seed)
    save = True
    use_gpu = True

    T = 55
    total_rate = 0.0
    step_interest_rate = (total_rate + 1) ** (1 / T) - 1

    S0 = 1
    V0 = 0.09
    drift = step_interest_rate
    kappa = 2.0
    theta = 0.04
    xi = 0.39
    rho = -0.75
    stock = HestonStock(S0, V0, drift, kappa, theta, xi, rho)

    contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility=0.4)
    hedging_instruments: List[Instrument] = [stock]
    cost_function = ProportionalCost(0.01)

    # CVaR 0.99
    criterion = RiskMeasures.CVaR(0.99) # CVaR with confidence level 0.99
    adam_weights = 'weights/cvar_099/Adam_weights.pt'
    sgld_weights = 'weights/cvar_099/SGLD_weights.pt'
    sghmc_weights = 'weights/cvar_099/SGHMC_weights.pt'

    # CVaR 0.5
    # criterion = RiskMeasures.CVaR(0.5) # CVaR with confidence level 0.5
    # adam_weights = 'weights/cvar_05/Adam_weights.pt'
    # sgld_weights = 'weights/cvar_05/SGLD_weights.pt'
    # sghmc_weights = 'weights/cvar_05/SGHMC_weights.pt'

    # Mean-Variance
    # criterion = RiskMeasures.ExpectationVariance(0.5) # Mean-Variance with risk aversion parameter 0.5
    # adam_weights = 'weights/mean_var/Adam_weights.pt'
    # sgld_weights = 'weights/mean_var/SGLD_weights.pt'
    # sghmc_weights = 'weights/mean_var/SGHMC_weights.pt'

    # # S&P 500
    prices_path = 'data/sp500.csv'
    file_prefix = 'outputs/validation/cvar_099'

    # Nasdaq
    # prices_path = 'data/nasdaq.csv'
    # file_prefix = 'outputs/validation/cvar_05'

    # Russell 2000
    # prices_path = 'data/russell2000.csv'
    # file_prefix = 'outputs/validation/mean_var'


    validate_from_weights(
        adam_weights,
        sgld_weights,
        sghmc_weights,
        prices_path,
        contingent_claim,
        hedging_instruments,
        criterion,
        step_interest_rate,
        cost_function,
        save,
        file_prefix,
        use_gpu
    )

if __name__ == "__main__":
    validation()


