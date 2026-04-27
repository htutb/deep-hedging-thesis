from typing import List
from matplotlib import pyplot as plt
import torch

from src.Costs import CostFunction, ProportionalCost, FixedCost
from src.agents.Agent import Agent
from src.agents.RecurrentAgent import RecurrentAgent
from src.agents.SimpleAgent import SimpleAgent
from src.agents.DeltaAgent import DeltaAgent
from src.agents.NakedAgent import NakedAgent

from src.instruments.Claims import Claim
from src.instruments.Derivatives import EuropeanCall, BSCall, BSPut, EuropeanPut
from src.instruments.Instruments import Instrument
from src.instruments.Primaries import GeometricBrownianStock, HestonStock, JumpStock

from src.optimizers import AdamOptimizer, SGLDOptimizer, SGHMCOptimizer

import src.RiskMeasures as RiskMeasures

from src.ExperimentRunner import ExperimentRunner, SimpleRunner, plot_dists


def main():
    seed = 133
    torch.manual_seed(seed)

    T = 55
    total_rate = 0.0
    step_interest_rate = (total_rate + 1) ** (1 / T) - 1

    # # Geometric Brownian Stock

    # S0 = 1
    # volatility = 0.2
    # drift = step_interest_rate
    # stock = GeometricBrownianStock(S0, drift, volatility)

    # Heston Stock

    S0 = 1
    V0 = 0.09
    drift = step_interest_rate
    kappa = 2.0
    theta = 0.04
    xi = 0.39
    rho = -0.75
    stock = HestonStock(S0, V0, drift, kappa, theta, xi, rho)

    contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility=0.30)
    hedging_instruments: List[Instrument] = [stock]

    epochs = 5000
    paths = int(1e5)
    verbose = True

    criterion = RiskMeasures.CVaR(0.99) # CVaR with confidence level 0.99
    # criterion2 = RiskMeasures.CVaR(0.99) # CVaR with confidence level 0.5

    prop_cost = 0.01
    cost_function: CostFunction = ProportionalCost(prop_cost)

    adam_optimizer = AdamOptimizer
    sgld_optimizer = SGLDOptimizer
    sghmc_optimizer = SGHMCOptimizer

    runners = []

    # # Delta hedging agent - requires strike, expiry, drift, volatility as stock_params
    # delta_params = (S0, T, drift, 0.75)  # strike, expiry, drift, volatility
    # delta_runner = SimpleRunner("Delta", use_gpu=True)
    # res = delta_runner.run(contingent_claim, hedging_instruments, criterion, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, stock_params=delta_params)
    # print(res)
    # runners.append(delta_runner)

    # # Simple agent - trainable, no stock_params needed
    # simple_runner = SimpleRunner("Simple", use_gpu=True)
    # h_dim = 16
    # res = simple_runner.run(contingent_claim, hedging_instruments, criterion, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    # print(res)
    # runners.append(simple_runner)

    # # Recurrent agent - trainable, no stock_params needed
    # recurrent_runner_adam = SimpleRunner("Recurrent", use_gpu=True)
    # h_dim = 16
    # res = recurrent_runner_adam.run(contingent_claim, hedging_instruments, criterion, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    # print(res)
    # runners.append(recurrent_runner_adam)

    # # Recurrent agent - trainable, no stock_params needed
    # recurrent_runner_adam = SimpleRunner("CVaR (p = 0.5)", use_gpu=True)
    # h_dim = 16
    # res = recurrent_runner_adam.run(contingent_claim, hedging_instruments, criterion, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    # print(res)
    # runners.append(recurrent_runner_adam)


    # # Recurrent agent - trainable, no stock_params needed
    # recurrent_runner_adam = SimpleRunner("CVaR (p = 0.99)", use_gpu=True)
    # h_dim = 16
    # res = recurrent_runner_adam.run(contingent_claim, hedging_instruments, criterion2, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    # print(res)
    # runners.append(recurrent_runner_adam)


    # Recurrent agent - trainable, no stock_params needed
    recurrent_runner_adam = SimpleRunner("Adam", use_gpu=True)
    h_dim = 1
    res = recurrent_runner_adam.run(contingent_claim, hedging_instruments, criterion, adam_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    print(res)
    runners.append(recurrent_runner_adam)

    # Recurrent agent - trainable, no stock_params needed
    recurrent_runner_sgld = SimpleRunner("SGLD", use_gpu=True)
    h_dim = 16
    res = recurrent_runner_sgld.run(contingent_claim, hedging_instruments, criterion, sgld_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    print(res)
    runners.append(recurrent_runner_sgld)

    # Recurrent agent - trainable, no stock_params needed
    recurrent_runner_sghmc = SimpleRunner("SGHMC", use_gpu=True)
    h_dim = 16
    res = recurrent_runner_sghmc.run(contingent_claim, hedging_instruments, criterion, sghmc_optimizer, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
    print(res)
    runners.append(recurrent_runner_sghmc)


    plot_dists(runners, save=True, file_prefix=f"outputs/adam_sgld_sghmc_0_99")
    plt.show()


if __name__ == "__main__":
    main()