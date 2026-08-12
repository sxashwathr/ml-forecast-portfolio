"""
monte_carlo.py
---------------
This is the FOURTH stage of the pipeline: instead of asking "what return do
we expect?" (which optimizer.py already answered), this file asks "what's
the RANGE of things that could realistically happen, and how bad could the
bad outcomes be?"

WHY this matters on top of the optimizer's single expected-return number:
A single expected return figure hides risk. A portfolio "expected" to
return 15%/year could still very plausibly lose 20% in a bad year -- the
expected value alone doesn't tell you that. Monte Carlo simulation
generates thousands of plausible future paths so we can see the full
distribution of outcomes, not just its average, and quantify the downside
with concrete risk metrics (Value at Risk, Expected Shortfall).

WHY we bootstrap from HISTORICAL returns instead of assuming returns are
normally distributed (a very common simpler approach):
Real stock returns have "fat tails" -- extreme days (crashes, rallies)
happen far more often than a normal distribution would predict. Sampling
actual historical daily returns (with replacement) instead of drawing from
a bell curve preserves whatever real skew and fat-tailedness existed in
this portfolio's actual history, which gives a more honest picture of
downside risk than a normal-distribution assumption would.

WHY np.random.seed(42) here specifically matters a lot: this file is the
most randomness-heavy part of the whole project (it draws hundreds of
thousands of random samples). Without fixing the seed, the VaR and
Expected Shortfall numbers would come out slightly different every run,
which would undermine trust in "the model says a 95% VaR of X%" -- a risk
number that changes every time you compute it isn't a very credible risk
number.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset
from optimization.optimizer import run_pipeline as run_optimizer

# WHY at least 5,000 simulations: Monte Carlo estimates (like our VaR and
# Expected Shortfall) get more stable/precise as we run more trials -- with
# too few, tail percentiles (like "the worst 5% of outcomes") are estimated
# from very few sample points and jump around a lot. 10,000 gives a solid
# margin above the requested minimum while still running in well under a
# second on a laptop, since this is fully vectorized with NumPy.
N_SIMULATIONS = 10_000
HORIZON_DAYS = 252  # WHY 252: one trading year, matching the "1-year horizon" the risk metrics are reported over
STARTING_VALUE = 10_000.0  # an arbitrary but intuitive starting portfolio value ($10,000) so results read like "if you invested $10k..."
VAR_CONFIDENCE = 0.95  # WHY 95%: the most common industry-standard confidence level for VaR reporting

FAN_CHART_PATH = os.path.join(os.path.dirname(__file__), "fan_chart.png")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "risk_summary.csv")


def compute_portfolio_historical_returns(features, weights):
    """
    Reconstruct what this portfolio's DAILY returns actually looked like
    historically, by applying the (fixed) target weights to each asset's
    actual historical daily returns.

    WHY we need this at all (rather than simulating each asset separately):
    the whole point of bootstrapping is to sample from a distribution that
    reflects reality -- including how assets have moved TOGETHER (their
    correlation). By first collapsing the multi-asset history down into a
    single historical portfolio-return series using the actual weights,
    a day where (say) NVDA crashed and SPY only dipped is preserved as one
    coherent portfolio-level data point, rather than being reconstructed
    from independently-sampled asset returns that would break the
    real historical correlation structure between assets.
    """
    returns_wide = features.pivot(index="date", columns="ticker", values="daily_return").dropna()
    returns_wide = returns_wide[weights.index]  # align column order to weights
    portfolio_returns = returns_wide @ weights.values
    return portfolio_returns


def simulate_paths(
    historical_portfolio_returns,
    n_simulations=N_SIMULATIONS,
    horizon_days=HORIZON_DAYS,
    starting_value=STARTING_VALUE,
    seed=42,
):
    """
    Generate `n_simulations` possible 1-year future portfolio value paths
    by resampling (with replacement) from the actual historical daily
    portfolio returns.

    WHY "with replacement" (bootstrap) instead of just replaying history
    once: we only have a few years of actual history -- far too little to
    directly show "the range of what could happen." Resampling WITH
    replacement lets us build thousands of alternate, equally plausible
    reshuffled year-long sequences from that same underlying pool of daily
    outcomes, which is exactly what the bootstrap technique is for.

    WHY this is a known simplification worth stating plainly: sampling each
    day independently ignores "volatility clustering" (the real-world
    tendency for calm days to follow calm days, and turbulent days to
    cluster together, e.g. around a crash). A fancier "block bootstrap"
    (resampling contiguous multi-day chunks instead of single days) would
    partially preserve that clustering. We use the simpler i.i.d. day-by-day
    version here because it's far easier to understand and explain, at the
    honest cost of understating how bad true crash clustering can get.
    """
    np.random.seed(seed)

    n_days_hist = len(historical_portfolio_returns)
    returns_array = historical_portfolio_returns.values

    # Draw a (n_simulations x horizon_days) grid of random day-indices, then
    # look up the actual historical return for each. Fully vectorized --
    # no Python-level loop over 10,000 simulations -- so this runs fast.
    sampled_indices = np.random.randint(0, n_days_hist, size=(n_simulations, horizon_days))
    sampled_returns = returns_array[sampled_indices]

    # Compound daily returns into a cumulative growth path: value on day t
    # is starting_value * (1+r1) * (1+r2) * ... * (1+rt). This is how
    # actual portfolio value compounds day over day.
    growth_factors = np.cumprod(1.0 + sampled_returns, axis=1)
    paths = starting_value * growth_factors

    # Prepend day-0 (the known starting value, before any simulated return)
    # so every path visually starts from the same point in the fan chart.
    day_zero = np.full((n_simulations, 1), starting_value)
    paths = np.hstack([day_zero, paths])
    return paths


def compute_risk_metrics(paths, starting_value=STARTING_VALUE, confidence=VAR_CONFIDENCE):
    """
    Turn the simulated paths into concrete, decision-useful risk numbers.

    WHY Value at Risk (VaR): it answers a very concrete question -- "at a
    95% confidence level, what's the worst loss I should expect over the
    next year?" It's one of the most widely used risk metrics in finance
    precisely because it's easy to state in plain English.

    WHY Expected Shortfall in addition to VaR (not instead of): VaR tells
    you the threshold of the bad 5% of outcomes, but says nothing about how
    bad THOSE outcomes are on average -- a portfolio could have the same
    VaR as another but a far worse "tail" beyond it. Expected Shortfall
    (a.k.a. Conditional VaR) averages the losses that occur BEYOND the VaR
    threshold, which is why regulators increasingly prefer it -- it
    doesn't ignore how severe the worst outcomes can get.
    """
    terminal_values = paths[:, -1]
    terminal_returns = terminal_values / starting_value - 1.0

    var_percentile = (1 - confidence) * 100  # e.g. 95% confidence -> the 5th percentile of outcomes
    value_at_risk = -np.percentile(terminal_returns, var_percentile)

    tail_mask = terminal_returns <= np.percentile(terminal_returns, var_percentile)
    expected_shortfall = -terminal_returns[tail_mask].mean()

    probability_of_loss = float(np.mean(terminal_returns < 0))

    return {
        "starting_value": starting_value,
        "median_terminal_value": float(np.median(terminal_values)),
        "mean_terminal_return": float(terminal_returns.mean()),
        f"VaR_{int(confidence*100)}": float(value_at_risk),
        f"expected_shortfall_{int(confidence*100)}": float(expected_shortfall),
        "probability_of_loss_1yr": probability_of_loss,
        "best_case_return": float(terminal_returns.max()),
        "worst_case_return": float(terminal_returns.min()),
    }


def plot_fan_chart(paths, save_path=FAN_CHART_PATH):
    """
    Visualize the simulation as a "fan chart": a median path surrounded by
    shaded percentile bands that widen further into the future.

    WHY a fan chart (rather than, say, plotting all 10,000 lines): plotting
    every path would be an unreadable mess. A fan chart instead shows the
    SHAPE of the uncertainty -- the bands naturally widen over time because
    compounding uncertainty over more days means outcomes spread out more
    the further into the future you look, which is an intuitive, honest way
    to communicate "we're much more sure about tomorrow than about a year
    from now."
    """
    days = np.arange(paths.shape[1])
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.fill_between(days, p5, p95, alpha=0.2, color="steelblue", label="5th-95th percentile")
    ax.fill_between(days, p25, p75, alpha=0.35, color="steelblue", label="25th-75th percentile")
    ax.plot(days, p50, color="navy", linewidth=2, label="Median path")
    ax.axhline(paths[0, 0], color="gray", linestyle="--", linewidth=1, label="Starting value")

    ax.set_xlabel("Trading Days Ahead")
    ax.set_ylabel("Simulated Portfolio Value ($)")
    ax.set_title(f"Monte Carlo Simulation: {paths.shape[0]:,} Bootstrapped 1-Year Portfolio Paths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_pipeline():
    """Full Monte Carlo pipeline, callable from app.py or other scripts."""
    _, features = load_dataset()
    optimizer_result = run_optimizer()
    weights = optimizer_result["max_sharpe_weights"]

    historical_returns = compute_portfolio_historical_returns(features, weights)
    paths = simulate_paths(historical_returns)
    metrics = compute_risk_metrics(paths)

    return {
        "weights": weights,
        "historical_returns": historical_returns,
        "paths": paths,
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = run_pipeline()
    metrics = result["metrics"]

    print(f"Simulated {N_SIMULATIONS:,} bootstrapped 1-year paths for the max-Sharpe portfolio")
    print(f"Starting value: ${metrics['starting_value']:,.0f}")
    print(f"Median terminal value after 1 year: ${metrics['median_terminal_value']:,.0f}")
    print(f"Mean expected 1-year return: {metrics['mean_terminal_return']:.2%}")
    print(f"\nValue at Risk (95% confidence):        {metrics['VaR_95']:.2%}")
    print(f"  -> Interpretation: 95% of the time, the 1-year loss should not exceed this.")
    print(f"Expected Shortfall (95% confidence):   {metrics['expected_shortfall_95']:.2%}")
    print(f"  -> Interpretation: IF we land in the worst 5% of outcomes, this is the average loss.")
    print(f"Probability of losing money over 1 year: {metrics['probability_of_loss_1yr']:.2%}")
    print(f"\nBest simulated 1-year return:  {metrics['best_case_return']:.2%}")
    print(f"Worst simulated 1-year return: {metrics['worst_case_return']:.2%}")

    plot_path = plot_fan_chart(result["paths"])
    print(f"\nSaved fan chart to {plot_path}")

    pd.Series(metrics).to_csv(SUMMARY_PATH, header=["value"])
    print(f"Saved risk summary to {SUMMARY_PATH}")
