from typing import List
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
from src.instruments.Instruments import Instrument

from src.optimizers import AdamOptimizer, SGLDOptimizer, SGHMCOptimizer


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


    def plot_path(self):
        portfolio_logs = self.agent.portfolio_logs
        # all quantities averaged across the  batch
        portfolio_value = portfolio_logs["portfolio_value"].mean(dim=0)   # (T,)
        cash_account    = portfolio_logs["cash_account"].mean(dim=0)      # (T,)
        claim_payoff    = portfolio_logs["claim_payoff"].mean()           
        claim_price     = claim_payoff

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for a in ax.flatten():
            a.grid(True)

        #left: mean P&L over time
        pnl = portfolio_value + cash_account
        pnl += claim_price
        pnl[-1] -= claim_payoff
        pnl = pnl[:-1]

        sns.lineplot(x=range(len(pnl)), y=pnl, ax=ax[0])
        ax[0].set_title("Mean Total P&L (Including Claim)")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("P&L")
        ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        max_val = max(abs(pnl.min()), abs(pnl.max())) * 1.1
        ax[0].set_ylim(-max_val, max_val)

        #right: mean terminal breakdown
        final_cash  = cash_account[-1] + claim_price
        final_pv    = portfolio_value[-1]
        categories  = ['Cash', 'PV', 'CC', 'P&L']
        values      = [final_cash.item(), final_pv.item(), -claim_payoff.item(),
                       (final_cash + final_pv - claim_payoff).item()]
        colors      = ['red', 'blue', 'green', 'orange']

        ax[1].bar(categories, values, color=colors)
        ax[1].set_title("Mean Final Portfolio Value Breakdown")
        ax[1].set_xlabel("Category")
        ax[1].set_ylabel("Value")
        ax[1].hlines(0, -1, len(categories), colors='black', linestyles='dashed')

        max_val = max(abs(min(values)), abs(max(values))) * 1.2
        ax[1].set_ylim(-max_val, max_val)

        padding = max_val * 0.04
        for j, v in enumerate(values):
            y  = v + padding if v >= 0 else v - padding
            va = 'bottom' if v >= 0 else 'top'
            ax[1].text(j, y, f"{v:.2f}", color='black', ha='center', va=va, fontweight='bold')

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
            self.plot_path()
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


def plot_positions_comparison(runners: List[ExperimentRunner]):
    """
    Plot all agents' mean hedge positions on a single axis with the BS delta as reference.
    If batchsize > 1, plot with +- 1std, else plot an exact position curve only.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(True)

    delta_plotted = False

    for runner in runners:
        portfolio_logs = runner.agent.portfolio_logs
        positions = portfolio_logs["positions"]          # (B, T, N)
        hedge_names = [h.__class__.__name__ for h in runner.hedging_instruments]

        mean_pos = positions.mean(dim=0)[:-1]            # (T-1, N)
        std_pos = positions.std(dim=0)[:-1]              # 0 if B=1

        for n in range(mean_pos.shape[1]):
            label = runner.agent_type if mean_pos.shape[1] == 1 else f"{runner.agent_type} — {hedge_names[n]}"
            t = range(mean_pos.shape[0])
            line = sns.lineplot(x=t, y=mean_pos[:, n], ax=ax, label=label)
            color = line.get_lines()[-1].get_color()
            if positions.shape[0] > 1:
                ax.fill_between(
                    t,
                    (mean_pos[:, n] - std_pos[:, n]).numpy(),
                    (mean_pos[:, n] + std_pos[:, n]).numpy(),
                    alpha=0.15,
                    color=color,
                )

        if not delta_plotted:
            claim_delta = portfolio_logs.get("claim_delta")
            if claim_delta is not None:
                mean_delta = claim_delta.mean(dim=0)[:-1]  # (T-1,)
                sns.lineplot(
                    x=range(len(mean_delta)),
                    y=mean_delta,
                    ax=ax,
                    color='red',
                    linestyle='-.',
                    alpha=0.9,
                    label="BS delta",
                )
                delta_plotted = True

    ax.set_title("Agent Positions vs BS Delta")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def validate_from_weights(
        adam_weights: str,
        sgld_weights: str,
        sghmc_weights: str,
        prices_path: str,
        contingent_claim: Claim,
        hedging_instruments: List[Instrument],
        criterion: torch.nn.Module,
        step_interest_rate: float = 0.0,
        cost_function: CostFunction = ProportionalCost(0.01),
        save: bool = True,
        file_prefix: str = "outputs/validation",
        use_gpu: bool = True,
) -> List[ExperimentRunner]:
    """
    Load pre-trained recurrent agent weights and run inference on real market data.

    Reads price data from a CSV file (each row is one price path, each column
    is one timestep — no header, no date column) and runs inference for each
    of the three optimisers (Adam, SGLD, SGHMC).
    """

    # expected csv format: no headers
    df = pd.read_csv(prices_path, header=None)
    primary_path = torch.tensor(df.values, dtype=torch.float32)  # (B, T) or (1, T)

    _, T = primary_path.shape

    weight_map = {
        "Adam":  (adam_weights,  AdamOptimizer),
        "SGLD":  (sgld_weights,  SGLDOptimizer),
        "SGHMC": (sghmc_weights, SGHMCOptimizer),
    }

    runners = []

    for agent_type, (weight_path, optimizer_cls) in weight_map.items():
        # load state dict (for h_dim)
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        h_dim = state_dict["network.fc1.weight"].shape[0]

        runner = SimpleRunner(agent_type, use_gpu=use_gpu)
        runner.agent = agents[agent_type](
            criterion,
            optimizer_cls,
            cost_function,
            hedging_instruments,
            step_interest_rate,
            h_dim=h_dim,
            use_gpu=use_gpu,
        )
        runner.agent.load_state_dict(state_dict)
        runner.agent.eval()

        device = runner.agent.device
        primary_path_dev = primary_path.to(device)

        with torch.no_grad():
            hedge_paths = torch.stack(
                [instr.value(primary_path_dev).to(device) for instr in hedging_instruments],
                dim=-1,
            )  # (B, T, N)
            runner.agent.compute_portfolio(hedge_paths, logging=True)

            claim_payoff = contingent_claim.payoff(primary_path_dev).to(device) # (B,)
            claim_delta = contingent_claim.delta(primary_path_dev) # (B, T)

            runner.agent.portfolio_logs["claim_payoff"] = claim_payoff.detach().cpu()
            runner.agent.portfolio_logs["claim_delta"] = (
                claim_delta.detach().cpu() if claim_delta is not None else None
            )

            # realized P&L: final PV + cash - claim
            portfolio_final = (
                runner.agent.portfolio_logs["portfolio_value"][:, -1]
                + runner.agent.portfolio_logs["cash_account"][:, -1]
            )  # (B,)
            profit = portfolio_final - claim_payoff.cpu()  # (B,)

            # risk-measure over the batch
            realized_loss = -criterion(profit).item()

        runner.portfolio_logs = runner.agent.portfolio_logs
        runner.claim = contingent_claim
        runner.hedging_instruments = hedging_instruments

        print(
            f"[{agent_type}]  mean P&L: {profit.mean():.4f}  std: {profit.std():.4f}"
            f"  |  Realized {criterion.__class__.__name__}: {realized_loss:.4f}"
        )
        runners.append(runner)

    output_dir = os.path.dirname(file_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # one summary plot per agent
    for runner in runners:
        fig = runner.plot_path()
        if save:
            fig.savefig(f"{file_prefix}_{runner.agent_type}_summary.pdf")

    # combined positions comparison
    fig_cmp = plot_positions_comparison(runners)
    if save:
        fig_cmp.savefig(f"{file_prefix}_positions_comparison.pdf")

    plt.show()

    return runners


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
    
    # CVaR (0.99)
    ax2.set_xlim(-0.2, 0.8)

    # CVaR (0.5)
    # ax2.set_xlim(-0.6, 0.4)

    # Mean-Variance
    # ax2.set_xlim(-0.6, 0.4)

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
    # CVaR (0.99)
    ax3.set_ylim(0, 5)

    # CVaR (0.5)
    # ax3.set_ylim(0, 3)

    # Mean-Variance
    # ax3.set_ylim(0, 3)

    if save:
        plt.savefig(f"{file_prefix}_training_losses.pdf")

    return ax1, ax2, ax3
