from typing import List
from matplotlib import pyplot as plt
import torch

from Costs import CostFunction, ProportionalCost
from instruments.Claims import Claim
from instruments.Derivatives import BSCall, BSPut
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock, HestonStock, JumpStock
import RiskMeasures
from ExperimentRunner import ExperimentRunner

seed = 1337
torch.manual_seed(seed)

T = 31
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1

# # Geometric Brownian Stock

# S0 = 1
# volatility = 0.2
# drift = step_interest_rate
# stock = GeometricBrownianStock(S0, drift, volatility)

# Heston Stock

S0 = 1
V0 = 0.04
drift = step_interest_rate
kappa = 1.0
theta = 0.04
xi = 0.30
pho = -0.7
stock = HestonStock(S0, V0, drift, kappa, theta, xi, pho)

# Jump Stock

# S0 = 1
# volatility = 0.2
# drift = step_interest_rate
# lam = 0.083
# m = -0.08
# v = 0.15
# stock = JumpStock(S0, drift, volatility, lam, m, v)

contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility=0.2)
hedging_instruments: List[Instrument] = [stock]

epochs = 50
paths = int(1e5)
verbose = True

criterion = RiskMeasures.CVaR(0.95) # CVaR with confidence level 0.95
# criterion = RiskMeasures.CVaR(0.99) # CVaR with confidence level 0.99

prop_cost = 0.01
cost_function: CostFunction = ProportionalCost(prop_cost)


recurrent_runner = ExperimentRunner("recurrent", use_gpu=True)
h_dim = 15
res = recurrent_runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
opt = recurrent_runner.agent.optimizer
print(res)
filename = f"output/{recurrent_runner.agent_type}_hd_{h_dim}_e_{epochs}_p_{paths}_s_{seed}_pc_{prop_cost: .1f}_opt_{opt.name}"
recurrent_runner.plot_runner(animate=True, save=True, file_prefix=filename, n=2)
