# deep-hedging-thesis
This thesis investigates the integration of Langevin-based optimizers into the Deep Hedging framework developed by Buehler et al., with the goal of improving convergence properties, escaping local minima, and ultimately pushing the performance boundaries of learned hedging strategies.


## How to run

```bash
git clone https://github.com/htutb/deep-hedging-thesis.git
cd deep-hedging-thesis
pip install -r requirements.txt
```

### Training

Edit `src/train.py` to select the risk measure (`CVaR(0.99)`, `CVaR(0.5)`, or `ExpectationVariance`) and which agents to train (Adam, SGLD, SGHMC). Weights are saved to `weights/`.

```bash
python -m src.train
```

### Validation on real market data

Place price CSVs (rows = paths, columns = timesteps, no header) in `data/`. Then edit `src/validation.py` to set `loss_name` (`"cvar_099"`, `"cvar_05"`, `"mean_var"`) and `index_name` (`"sp500"`, `"nasdaq"`, `"russell2000"`). Results and plots are saved to `validation/{loss_name}/{index_name}/`.

```bash
python -m src.validation
```