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

    contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility=0.3)
    hedging_instruments: List[Instrument] = [stock]
    cost_function = ProportionalCost(0.01)

    # loss function 
    # "cvar_099" "cvar_05" "mean_var"
    loss_name = "mean_var"

    loss_configs = {
        "cvar_099": {
            "criterion":     RiskMeasures.CVaR(0.99),
            "adam_weights":  "weights2/cvar_099/Adam_weights.pt",
            "sgld_weights":  "weights2/cvar_099/SGLD_weights.pt",
            "sghmc_weights": "weights2/cvar_099/SGHMC_weights.pt",
        },
        "cvar_05": {
            "criterion":     RiskMeasures.CVaR(0.5),
            "adam_weights":  "weights2/cvar_05/Adam_weights.pt",
            "sgld_weights":  "weights2/cvar_05/SGLD_weights.pt",
            "sghmc_weights": "weights2/cvar_05/SGHMC_weights.pt",
        },
        "mean_var": {
            "criterion":     RiskMeasures.ExpectationVariance(1.0),
            "adam_weights":  "weights/mean_var/Adam_weights.pt",
            "sgld_weights":  "weights/mean_var/SGLD_weights.pt",
            "sghmc_weights": "weights/mean_var/SGHMC_weights.pt",
        },
    }

    criterion     = loss_configs[loss_name]["criterion"]
    adam_weights  = loss_configs[loss_name]["adam_weights"]
    sgld_weights  = loss_configs[loss_name]["sgld_weights"]
    sghmc_weights = loss_configs[loss_name]["sghmc_weights"]

    # index
    # "sp500" / "nasdaq" / "russell2000"
    index_name = "russell2000"

    index_configs = {
        "sp500":      "data/sp500.csv",
        "nasdaq":     "data/nasdaq.csv",
        "russell2000": "data/russell2000.csv",
    }

    prices_path = index_configs[index_name]
    file_prefix = f"validation/{loss_name}/{index_name}"


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


