from typing import List
from matplotlib import pyplot as plt
import torch

from Costs import CostFunction, ProportionalCost, FixedCost
from agents.RecurrentAgent import RecurrentAgent
from agents.SimpleAgent import SimpleAgent
from instruments.Claims import Claim
from instruments.Derivatives import EuropeanCall, BSCall, BSPut, EuropeanPut
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock, HestonStock
import RiskMeasures
from ExperimentRunner import ExperimentRunner, SimpleRunner, plot_dists

seed = 1337
torch.manual_seed(seed)

T = 31 # Number of time steps
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1

###### Define an underlying stock

# Geometric Brownian Stock

S0 = 1
volatility = 0.2
drift = step_interest_rate
stock = GeometricBrownianStock(S0, drift, volatility)

# Heston Stock

# S0 = 1
# V0 = 0.04
# drift = step_interest_rate
# kappa = 1.0
# theta = 0.04
# xi = 0.30
# pho = -0.7
# stock = HestonStock(S0, V0, drift, kappa, theta, xi, pho)

# Jump Stock

# S0 = 1
# volatility = 0.2
# drift = step_interest_rate
# lam = 0.083
# m = -0.08
# v = 0.15
# stock = JumpStock(S0, drift, volatility, lam, m, v)


# Define the claim, hedging instruments, criterion, and costs for the experiment
contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [stock]

criterion = RiskMeasures.CVaR(0.95) # CVaR with confidence level 0.95
# criterion = RiskMeasures.CVaR(0.99) # CVaR with confidence level 0.99


cost_function: CostFunction = ProportionalCost(0.0)
# cost_function: CostFunction = ProportionalCost(0.01)


# Define the agent
epochs = 50
h_dim = 15
paths = int(1e5)
verbose = True

# Choose which agent to run
# runner = ExperimentRunner("recurrent", use_gpu=True)

# For delta agent, use SimpleRunner with stock_params
runner = SimpleRunner("delta", use_gpu=True)
delta_params = (S0, T, drift, volatility)  # strike, expiry, drift, volatility

# Run the experiment
runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim, stock_params=delta_params)

# Plot the results
runner.plot_runner(animate=True, save=True, n=3)
