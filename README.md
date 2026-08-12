# ML Forecast Portfolio

An end-to-end, beginner-documented portfolio optimization pipeline: a Ridge
regression model forecasts each asset's next-period return, an SLSQP
optimizer builds a maximum-Sharpe-ratio portfolio from those forecasts, a
Monte Carlo simulation stress-tests that portfolio's downside risk, and a
walk-forward backtest **honestly checks whether the ML model actually
helped** — rather than assuming it did.

Every source file has extensive inline comments explaining not just *what*
each step does, but *why* it exists and why it's built the way it is. See
[`EXPLAINED.md`](EXPLAINED.md) for a plain-English, interview-ready
walkthrough of the whole project.

## Project summary

The universe is 10 tickers: 8 robotics/automation/deep-tech names
(`ISRG`, `ROK`, `PH`, `TER`, `CGNX`, `NVDA`, `EMR`, `HON`) plus 2 broad-market
ETFs (`SPY`, `QQQ`) used as a sanity-check baseline. The pipeline:

1. **Pull data** — 5+ years of daily prices, plus rolling volatility and
   momentum features.
2. **Forecast returns** — a per-ticker Ridge regression predicts each
   asset's next-day return from its recent volatility/momentum, using a
   strictly time-based train/test split (no lookahead bias).
3. **Optimize** — `scipy.optimize.minimize(method='SLSQP')` finds the
   long-only, fully-invested portfolio weights that maximize the Sharpe
   ratio, and traces out the efficient frontier.
4. **Simulate risk** — 10,000 bootstrapped 1-year paths quantify Value at
   Risk, Expected Shortfall, and probability of loss for that portfolio.
5. **Backtest, honestly** — a walk-forward test (monthly rebalancing, 2-year
   rolling estimation window) compares three strategies — equal-weight,
   optimized-on-naive-historical-mean, and optimized-on-ML-predictions — to
   see whether the ML model's forecasts actually earned their complexity.
6. **Dashboard** — a Streamlit app ties it all together with a risk-tolerance
   slider that picks a point on the efficient frontier.

Reproducibility: `np.random.seed(42)` is set everywhere randomness is used
(the ML train/test tooling and the Monte Carlo bootstrap), so re-running the
pipeline on any machine reproduces the same numbers.

## Folder structure

```
ml-forecast-portfolio/
├── data/
│   └── data_loader.py       # Pulls prices, computes return/volatility/momentum features
├── models/
│   └── return_predictor.py  # Ridge regression, time-based split, per-ticker forecasts
├── optimization/
│   └── optimizer.py         # SLSQP max-Sharpe optimizer + efficient frontier
├── simulation/
│   └── monte_carlo.py       # Bootstrapped Monte Carlo simulation, VaR, Expected Shortfall
├── backtest/
│   └── backtest.py          # Walk-forward backtest: equal-weight vs. naive-mean vs. ML
├── app.py                   # Streamlit dashboard
├── requirements.txt
├── README.md
└── EXPLAINED.md             # Plain-English walkthrough of the whole project
```

Each script also caches its output (CSVs and PNGs, alongside the script) so
downstream stages — and the dashboard — don't need to recompute everything
from scratch every time.

## Setup instructions

```bash
# 1. Clone the repo
git clone https://github.com/sxashwathr/ml-forecast-portfolio.git
cd ml-forecast-portfolio

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline stages in order (each prints its results and caches
#    its output for the next stage)
python data/data_loader.py
python models/return_predictor.py
python optimization/optimizer.py
python simulation/monte_carlo.py
python backtest/backtest.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

> **Note on Python version:** this project was built and tested on Python
> 3.13. If `python3 -m venv` fails with a `pyexpat`/`ensurepip` error on
> macOS, it's a broken Homebrew Python bottle unrelated to this project —
> try a different installed Python version (e.g. `python3.13 -m venv venv`).

## Screenshots

[SCREENSHOT: efficient frontier]

[SCREENSHOT: monte carlo fan chart]

[SCREENSHOT: dashboard]

## What I learned / limitations — did the ML actually help?

**Short answer: partially, and it's genuinely mixed.**

Walk-forward backtest results (Sharpe ratio, out-of-sample, 912 trading
days / ~3.6 years):

| Strategy | Cumulative Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| Equal-weight (baseline) | 188.6% | 1.22 | -28.4% |
| Optimized — naive historical mean | 660.7% | **1.58** | -28.7% |
| Optimized — ML-predicted returns | 589.5% | 1.56 | -26.9% |

The ML-driven strategy clearly beat doing nothing sophisticated
(equal-weight), which is a real result. But it did **not** beat the much
simpler "just optimize using the recent historical average return"
approach — naive-mean edged it out on both Sharpe ratio and cumulative
return. Given that `return_predictor.py` honestly showed an average test
R² of **-0.009** (essentially zero predictive power) and 50.8% directional
accuracy (a coin flip) back in step 2, this isn't surprising — it's the
expected, honest outcome of feeding a weak, noisy signal through an
optimizer.

That last part matters and is worth calling out directly: portfolio
optimizers are extremely sensitive to their expected-return inputs (a
well-known effect sometimes called "estimation-error maximization"). Feed
one a noisy signal and it will confidently, aggressively bet on the noise —
which is exactly what happened when the ML-based optimizer concentrated
83% of the "optimal" portfolio into a single stock (`HON`) based on
predictions with essentially no real skill behind them. The backtest is
what caught that a static "look, the optimizer found a Sharpe of 2.67
portfolio!" result would have hidden.

**Other limitations, honestly listed:**

- **No transaction costs or taxes** — monthly rebalancing in the backtest
  assumes free, instant trades, which flatters any strategy that turns over
  its holdings, including the naive-mean and ML strategies here.
- **Small, simple feature set** — only 30/90-day momentum and 30-day
  volatility feed the model. A more sophisticated project might add
  fundamental data, sentiment, or macro features — though more features
  don't guarantee more real signal.
- **i.i.d. bootstrap in the Monte Carlo simulation** — daily returns are
  resampled independently, which understates real-world "volatility
  clustering" (bad days tend to cluster around crashes). A block bootstrap
  would be more realistic.
- **Small, fixed universe** — 10 tickers, chosen for a specific theme
  (robotics/automation/deep-tech), is not a diversified enough universe for
  real investing; it's sized for a learning project.
- **Efficient Market Hypothesis context** — predicting next-day stock
  returns from public technical features alone is known to be extremely
  hard. The honest near-zero R² here is consistent with that, not a sign
  something was implemented incorrectly.

The most important lesson from this project isn't "ML beats/loses to
naive baselines" in the abstract — it's that **you only find out which is
true by testing honestly, out-of-sample, against a simple baseline** —
and that a good-looking optimizer output (high Sharpe, confident weights)
can be a red flag for overfitting to noisy inputs rather than evidence of
a good strategy.
