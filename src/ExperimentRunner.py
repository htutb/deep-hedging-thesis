from typing import List
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from matplotlib import pyplot as plt
import torch
import os

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

import src.RiskMeasures as RiskMeasures


agents = {
    "Simple": SimpleAgent,
    "Recurrent": RecurrentAgent,
    "CVaR (p = 0.5)": RecurrentAgent,
    "CVaR (p = 0.99)": RecurrentAgent,
    "Adam": RecurrentAgent,
    "SGLD": RecurrentAgent,
    "SGHMC": RecurrentAgent,
    "Delta": DeltaAgent,
    "Naked": NakedAgent
}

class ExperimentRunner:

    def __init__(self, agent_type: str, use_gpu=True) -> None:
        self.use_gpu = use_gpu
        self.agent_type = agent_type


    def run(self,
            contingent_claim: Claim,
            hedging_instruments: List[Instrument],
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            T = 31,
            step_interest_rate = 0.0,
            epochs = 50,
            paths = int(1e5),
            verbose = True,
            cost_function: CostFunction = ProportionalCost(0.01),
            h_dim = 15
            ) -> None:

        self.agent: Agent = agents[self.agent_type](criterion, optimizer, cost_function, hedging_instruments, step_interest_rate, h_dim=h_dim, use_gpu=self.use_gpu)

        # Only fit trainable agents (simple and recurrent have optimizer/scheduler)
        if self.agent_type in ["Simple", "CVaR (p = 0.5)", "CVaR (p = 0.99)", "Recurrent", "Adam", "SGLD", "SGHMC"]:
            self.agent.fit(contingent_claim, epochs, paths, verbose, T, logging=True)
            self.training_logs = self.agent.training_logs

        # All agents can validate
        loss = self.agent.validate(contingent_claim, int(1e6), T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        self.claim = contingent_claim
        self.hedging_instruments = hedging_instruments
        self.training_logs = self.agent.training_logs

        return loss


    def plot_val_dist(self, save=False, file_prefix='plot'):
        return plot_dists([self], save, file_prefix)

    def training_pnl_animation(self):
        training_pl = self.training_logs["training_PL"]
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"Training P&L, N: {len(training_pl[i])}, Epoch: {i+1}")
            ax.set_xlim(-4, 4)
            ax.grid()
            ax.set_xlabel("P&L")
            ax.set_ylabel("Frequency")
            ax.set_ylim(0, len(training_pl[i]) / 2)
            sns.histplot(training_pl[i].numpy(), ax=ax, stat='count', kde=False, color='blue', label='P&L', binwidth=0.1)
        return FuncAnimation(fig, animate, frames=len(training_pl), repeat=True)


    def plot_training_loss(self):
        losses = self.training_logs["training_losses"]
        plot = sns.lineplot(x=range(len(losses)), y=losses)
        plot.set_title("Training Loss")
        plot.set_xlabel("Epoch")
        plot.set_ylabel("Loss")
        plot.grid()
        plot.set_yscale('log')
        return plot


    def plot_path(self, i):
        portfolio_logs = self.agent.portfolio_logs
        portfolio_value = portfolio_logs["portfolio_value"][i]
        cash_account = portfolio_logs["cash_account"][i]
        positions = portfolio_logs["positions"][i]
        hedge_paths = portfolio_logs["hedge_paths"][i]
        claim_payoff = portfolio_logs["claim_payoff"][i]
        claim_price = portfolio_logs["claim_payoff"].mean()
        claim_delta = None
        claim_name = self.claim.__class__.__name__
        hedge_names = [h.__class__.__name__ for h in self.hedging_instruments]

        if "claim_delta" in portfolio_logs and portfolio_logs["claim_delta"] is not None:
            claim_delta = portfolio_logs["claim_delta"][i]

        fig, ax = plt.subplots(2, 2)

        for a in ax.flatten():
            a.grid(True)  # Adding grid to each subplot

        for i in range(hedge_paths.shape[1]):
            sns.lineplot(x=range(len(hedge_paths[:, i])), y=hedge_paths[:, i], ax=ax[0, 0], label=hedge_names[i])
        ax[0,0].set_title("Price of Hedging Instruments")
        ax[0,0].set_xlabel("Time")
        ax[0,0].set_ylabel("Price")
        ax[0,0].legend()


        for i in range(positions.shape[1]):
            sns.lineplot(x=range(len(positions[:, i])), y=positions[:, i], ax=ax[0, 1], label=hedge_names[i])
        ax[0, 1].set_title("Agent Positions")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Positions")
        # add line for y = 0
        ax[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        # make diverging from 0
        max_value = max(abs(positions.min()), abs(positions.max()))

        if claim_delta is not None:
            # add the delta of the claim to the graph
            ax[0, 1] = sns.lineplot(x=range(len(claim_delta)), y=claim_delta, ax=ax[0, 1], color='red', label=f'Delta of {claim_name}', alpha=0.9, linestyle='-.')
            max_delta = max(abs(claim_delta.min()), abs(claim_delta.max()))
            lim = max(max_value, max_delta) * 1.1
        else:
            lim = max_value * 1.1
        ax[0, 1].set_ylim(-lim, lim)


        pnl = portfolio_value + cash_account
        pnl += claim_price
        pnl[-1] -= claim_payoff

        sns.lineplot(x=range(len(pnl)), y=pnl, ax=ax[1, 0])
        ax[1, 0].set_title("Total P&L (Including Claim)")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("P&L")
        # add line for y = 0
        ax[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # make diverging from 0
        max_value = max(abs(pnl.min()), abs(pnl.max())) * 1.1
        ax[1, 0].set_ylim(-max_value, max_value)

        final_cash = cash_account[-1] + claim_price
        final_portfolio_value = portfolio_value[-1]

        # Setting up the diverging bars
        categories = ['Cash', 'PV', 'CC', 'P&L']
        values = [final_cash, final_portfolio_value, -claim_payoff, final_cash + final_portfolio_value -claim_payoff]
        colors = ['red', 'blue', 'green', 'orange']

        ax[1,1].bar(categories, values, color=colors)
        ax[1,1].set_title("Final Portfolio Value Breakdown")
        ax[1,1].set_xlabel("Category")
        ax[1,1].set_ylabel("Value [€]")

        # Adding a horizontal line at y=0
        ax[1,1].hlines(0, -1, len(categories), colors='black', linestyles='dashed')

        # Set y limits to center the plot around 0
        max_value = max(abs(min(values)), abs(max(values))) * 1.1
        ax[1,1].set_ylim(-max_value, max_value)

        # Adding value annotations to the bars
        for i, v in enumerate(values):
            ax[1,1].text(i, v, f" {v:.2f}", color='black', ha='center', fontweight='bold')

        # make sure the text does not overlap
        fig.tight_layout()

        return fig


    def plot_runner(self, animate=False, save=False, file_prefix='plot', n = 5, compare = []):
        self.plot_training_loss()
        if save:
            plt.savefig(f'{file_prefix}_training_loss.pdf')
        plt.show()

        if animate:
            ani = self.training_pnl_animation()
            if save:
                ani.save(f'{file_prefix}_training_animation.mp4', writer='ffmpeg')
            plt.show()


        plot_dists([*compare, self], save=save, file_prefix=file_prefix)

        for i in range(n):
            self.plot_path(i)
            if save:
                plt.savefig(f'{file_prefix}_path_{i}.pdf')
        plt.show()


class SimpleRunner(ExperimentRunner):
    """
    Runner specifically designed for agents with different constructor signatures.
    - Trainable agents (simple, recurrent): No stock_params needed
    - Fixed-strategy agents (delta, naked): stock_params required
    """

    def run(self,
            contingent_claim: Claim,
            hedging_instruments: List[Instrument],
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            T = 31,
            step_interest_rate = 0.0,
            epochs = 50,
            paths = int(1e5),
            verbose = True,
            cost_function: CostFunction = ProportionalCost(0.01),
            h_dim = 15,
            stock_params = None,
            ) -> None:

        # Handle different agent initialization signatures
        if self.agent_type in ["Simple", "CVaR (p = 0.5)", "CVaR (p = 0.99)", "Recurrent", "Adam", "SGLD", "SGHMC"]:
            # Trainable agents: need optimizer
            self.agent = agents[self.agent_type](criterion, optimizer, cost_function, hedging_instruments, step_interest_rate, h_dim=h_dim, use_gpu=self.use_gpu)
        else:  # delta, naked
            # Fixed-strategy agents: require stock_params
            if stock_params is None:
                raise ValueError(f"Agent type '{self.agent_type}' requires 'stock_params' argument")
            self.agent = agents[self.agent_type](criterion, cost_function, hedging_instruments, stock_params, step_interest_rate, h_dim=h_dim, use_gpu=self.use_gpu)

        # Only fit trainable agents (simple and recurrent have optimizer/scheduler)
        if self.agent_type in ["Simple", "CVaR (p = 0.5)", "CVaR (p = 0.99)", "Recurrent", "Adam", "SGLD", "SGHMC"]:
            self.agent.fit(contingent_claim, epochs, paths, verbose, T, logging=True)
            self.training_logs = self.agent.training_logs
            os.makedirs("weights", exist_ok=True)
            torch.save(self.agent.state_dict(), f"weights/{self.agent_type}_weights.pt")

        # All agents can validate
        loss = self.agent.validate(contingent_claim, int(1e6), T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        self.claim = contingent_claim
        self.hedging_instruments = hedging_instruments
        return loss

    def plot_runner(self, animate=False, save=False, file_prefix='plot', n=5, compare=[]):
        self.plot_val_dist()
        if save:
            plt.savefig(f'{file_prefix}_val_dist.pdf')
        plt.show()


def plot_dists(runners: List[ExperimentRunner], save=False, file_prefix='plot', x_lim=25):
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()

    for runner in runners:
        val_profit = runner.validation_logs["validation_profit"]
        
        sns.histplot(
            (val_profit).numpy(),
            binwidth=0.03,
            stat='density',
            kde=False,
            label=f'{runner.agent_type}',
            alpha=0.5,
            edgecolor=None,
            linewidth=0,
            ax=ax1
        )

    ax1.set_title("Hedging P&L")
    ax1.set_xlim(-1, 0.5)
    ax1.grid()
    ax1.legend()

    if save:
        plt.savefig(f'{file_prefix}_pnl.pdf')

    plt.figure(figsize=(12, 8))
    ax2 = plt.gca()

    for runner in runners:
        val_profit = runner.validation_logs["validation_profit"]

        price = runner.validation_logs["price"]
        hedging_error = val_profit + price
        realized_cvar = -runner.agent.criterion(hedging_error).item()

        sns.histplot(
            hedging_error.numpy(),
            binwidth=0.03,
            stat='density',
            kde=False,
            label=f'{runner.agent_type}, Price: {price:.4f}, Realized CVaR: {realized_cvar:.4f}',
            alpha=0.5,
            edgecolor=None,
            linewidth=0,
            ax=ax2
        )


    ax2.set_title("Price-adjusted P&L")
    # ax2.set_xlim(9, 11)
    ax2.set_xlim(-0.5, 1.5)
    ax2.grid()
    ax2.legend()

    if save:
        plt.savefig(f'{file_prefix}_price_adjusted_pnl.pdf')

    plt.figure(figsize=(12, 8))
    ax3 = plt.gca()

    for runner in runners:
        if hasattr(runner, "training_logs") and "training_losses" in runner.training_logs:
            losses = runner.training_logs["training_losses"].numpy()
            val_loss = runner.validation_logs["validation_loss"]

            ax3.plot(
                range(len(losses)), 
                losses, 
                label=f'{runner.agent_type}, Loss: {val_loss:.2f}'
            )

    ax3.set_title("Training Loss Curves")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True)
    ax3.legend()
    ax3.set_ylim(8, 40)

    if save:
        plt.savefig(f"{file_prefix}_training_losses.pdf")

    return ax1, ax2, ax3
