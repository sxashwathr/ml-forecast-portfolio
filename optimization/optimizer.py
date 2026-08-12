"""
optimizer.py
------------
This is the THIRD stage of the pipeline: given (a) an expected return per
asset -- from models/return_predictor.py -- and (b) how those assets have
historically moved together (their covariance), find the portfolio weights
that produce the best risk-adjusted return.

WHY "risk-adjusted" and not just "highest expected return":
A portfolio that dumps 100% into the single asset with the highest
predicted return would technically maximize expected return, but it would
also carry all of that asset's risk with zero diversification benefit.
The Sharpe ratio -- (portfolio return - risk-free rate) / portfolio
volatility -- rewards return per unit of risk taken, which is a far more
sensible thing to optimize for in a real portfolio.

WHY scipy.optimize.minimize with method='SLSQP' (Sequential Least Squares
Programming) instead of, say, a closed-form formula:
Maximizing the Sharpe ratio subject to "weights are between 0 and 1" and
"weights sum to 1" is a constrained nonlinear optimization problem -- there
isn't a simple closed-form solution once those bounds are added (there IS
a closed-form solution for the *unconstrained* max-Sharpe portfolio, but it
can produce negative weights, i.e. short-selling, which we've explicitly
disallowed). SLSQP is a standard, well-tested algorithm for exactly this
kind of "smooth objective + smooth equality/inequality constraints"
problem.

WHY no np.random.seed() call is needed in this file: SLSQP is a
deterministic local optimizer -- given the same starting point (we always
start from equal weights) and the same inputs, it always converges to the
same answer. There's no randomness to fix here, unlike the ML train/test
split or the Monte Carlo simulation.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # WHY: lets this script save plots to a file without needing a GUI/display backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset
from models.return_predictor import run_pipeline as run_return_predictor

# WHY 4%: a rough, round approximation of a short-term US Treasury yield.
# The Sharpe ratio needs SOME risk-free rate to represent "the return you
# could get with effectively zero risk" -- without subtracting it, the
# ratio would reward volatility-adjusted return even against an asset that
# barely outperforms holding cash, which overstates how good that
# portfolio really is.
RISK_FREE_RATE_ANNUAL = 0.04

# WHY 252: the standard approximate number of US stock market trading days
# in a year. We use it consistently to convert our daily numbers (daily
# predicted return, daily covariance) into annualized ones, which is the
# convention Sharpe ratios and portfolio risk are normally quoted in.
TRADING_DAYS_PER_YEAR = 252

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "optimal_weights.csv")
FRONTIER_PATH = os.path.join(os.path.dirname(__file__), "efficient_frontier.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "efficient_frontier.png")


def build_annualized_inputs(predicted_daily_returns, features):
    """
    Convert the ML model's DAILY predicted returns and the assets'
    historical DAILY covariance into ANNUALIZED versions.

    WHY annualize with simple multiplication (mu * 252, cov * 252) instead
    of compounding: daily returns here are tiny (fractions of a percent),
    and over the short daily-to-annual conversion, simple scaling is a
    standard, easy-to-explain approximation that's accurate enough for
    portfolio construction. This keeps the math transparent for a reader
    who isn't a quant, at a negligible cost in precision.

    WHY the covariance matrix comes from HISTORICAL returns, not the ML
    model: the model predicts a single expected return per asset, but it
    doesn't predict how assets will co-move with each other. Historical
    covariance is a reasonable, standard estimate of that -- how NVDA and
    QQQ have actually moved together in the past is a much more grounded
    basis for a risk estimate than trying to derive it from a weak 3-feature
    regression.
    """
    mu_annual = predicted_daily_returns * TRADING_DAYS_PER_YEAR

    # Pivot the long feature table into a wide Date x Ticker table of daily
    # returns, aligned so every row is a date all assets have data for --
    # essential, because covariance requires comparing assets on the SAME
    # days.
    returns_wide = features.pivot(index="date", columns="ticker", values="daily_return")
    returns_wide = returns_wide[mu_annual.index]  # keep column order consistent with mu
    returns_wide = returns_wide.dropna()

    cov_annual = returns_wide.cov() * TRADING_DAYS_PER_YEAR
    return mu_annual, cov_annual


def portfolio_performance(weights, mu, cov):
    """
    Given a set of weights, compute the portfolio's expected annual return,
    annual volatility (standard deviation), and Sharpe ratio.

    WHY volatility uses the full covariance matrix (w' * Cov * w) rather
    than a weighted average of individual volatilities: a weighted AVERAGE
    of volatilities ignores diversification entirely -- it assumes assets
    move independently. The quadratic form w' * Cov * w correctly accounts
    for the fact that if two assets move in opposite directions, holding
    both REDUCES portfolio risk below what a simple average would suggest.
    That correction is the entire mathematical reason diversification works.
    """
    weights = np.asarray(weights)
    port_return = float(weights @ mu)
    port_variance = float(weights @ cov.values @ weights)
    port_volatility = float(np.sqrt(port_variance))
    sharpe = (port_return - RISK_FREE_RATE_ANNUAL) / port_volatility
    return port_return, port_volatility, sharpe


def _negative_sharpe(weights, mu, cov):
    # SLSQP only minimizes, so to MAXIMIZE Sharpe ratio we minimize its
    # negative -- a standard trick, not a mathematical subtlety.
    return -portfolio_performance(weights, mu, cov)[2]


def optimize_max_sharpe(mu, cov):
    """
    Find the portfolio weights that maximize the Sharpe ratio, subject to:
      - weights sum to 1 (fully invested, no leftover/borrowed cash)
      - each weight is between 0 and 1 (no short-selling, no leverage --
        a beginner-friendly, "long-only" constraint that also matches how
        most retail investors actually can allocate)
    """
    n_assets = len(mu)
    x0 = np.repeat(1.0 / n_assets, n_assets)  # WHY start at equal weights: an unbiased, neutral starting point that doesn't presuppose any asset is better
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        _negative_sharpe,
        x0,
        args=(mu, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed to converge: {result.message}")

    weights = pd.Series(result.x, index=mu.index, name="weight")
    # Numerical solvers can leave tiny negative residues like -1e-17 instead
    # of exactly 0, or drift a hair outside bounds. Clipping and
    # renormalizing keeps the output a clean, valid set of weights.
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()
    return weights


def minimize_volatility_for_target_return(mu, cov, target_return):
    """
    Find the lowest-volatility portfolio that achieves AT LEAST a given
    target return. Used to trace out the efficient frontier: for many
    target returns, find the minimum risk needed to achieve each one.

    WHY this is the standard way to build an efficient frontier: the
    frontier is defined as "the set of portfolios where you can't get more
    return without taking more risk." Sweeping target returns and solving
    for minimum volatility at each one traces exactly that boundary.
    """
    n_assets = len(mu)
    x0 = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu.values - target_return},
    ]

    def portfolio_volatility(w):
        return np.sqrt(w @ cov.values @ w)

    result = minimize(
        portfolio_volatility,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        return None
    return result.x


def compute_efficient_frontier(mu, cov, n_points=50):
    """
    Sweep a range of achievable target returns (from the lowest-return
    asset to the highest-return asset) and, for each, find the minimum-risk
    portfolio. Returns a DataFrame describing the frontier plus the
    weights at each point.

    WHY we filter down to the "upper" branch before returning: minimizing
    volatility for a FIXED target return produces a full parabola-shaped
    curve in risk/return space -- for very low target returns, achieving
    them exactly can actually require MORE risk than the minimum-variance
    portfolio needs (imagine being forced to hold a low-return, high-risk
    asset just to hit a low target exactly). That lower branch is real
    output of the math, but it's not "efficient" by the standard
    definition -- a rational investor would never accept its risk level
    for that little return, since the minimum-variance portfolio alone
    offers less risk AND more return. Keeping only points at or above the
    global-minimum-variance portfolio's return is what makes this
    genuinely "the efficient frontier" rather than the full minimum-
    variance curve, and it matters beyond the plot: app.py maps a
    risk-tolerance slider onto the nearest point on this frontier by
    volatility alone, and without this filter it could silently hand back
    a dominated, worse-than-necessary portfolio.
    """
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    rows = []
    for target in target_returns:
        weights = minimize_volatility_for_target_return(mu, cov, target)
        if weights is None:
            continue  # WHY skip: some extreme target returns near the boundary can be numerically infeasible to hit exactly; skipping them is safer than reporting a wrong number
        ret, vol, sharpe = portfolio_performance(weights, mu, cov)
        rows.append({"target_return": target, "return": ret, "volatility": vol, "sharpe": sharpe,
                      **{f"weight_{t}": w for t, w in zip(mu.index, weights)}})

    frontier = pd.DataFrame(rows)
    min_variance_idx = frontier["volatility"].idxmin()
    min_variance_return = frontier.loc[min_variance_idx, "return"]
    efficient_frontier = frontier[frontier["return"] >= min_variance_return].sort_values("volatility").reset_index(drop=True)
    return efficient_frontier


def plot_efficient_frontier(frontier_df, mu, cov, max_sharpe_weights, save_path=PLOT_PATH):
    """
    Visualize the efficient frontier alongside individual assets and the
    max-Sharpe portfolio.

    WHY show individual assets on the same chart: it's the clearest way to
    SEE diversification working -- the frontier curve should sit up and to
    the left of most individual assets, meaning a well-constructed
    portfolio gets more return per unit of risk than almost any single
    asset alone.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(frontier_df["volatility"], frontier_df["return"], "b-", linewidth=2, label="Efficient Frontier")

    for ticker in mu.index:
        asset_return = mu[ticker]
        asset_vol = np.sqrt(cov.loc[ticker, ticker])
        ax.scatter(asset_vol, asset_return, marker="o", s=60)
        ax.annotate(ticker, (asset_vol, asset_return), xytext=(5, 5), textcoords="offset points", fontsize=9)

    max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = portfolio_performance(max_sharpe_weights, mu, cov)
    ax.scatter(max_sharpe_vol, max_sharpe_return, marker="*", s=400, color="gold",
               edgecolor="black", zorder=5, label=f"Max Sharpe Portfolio (Sharpe={max_sharpe_ratio:.2f})")

    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Annualized Expected Return")
    ax.set_title("Efficient Frontier: ML-Predicted Returns vs. Historical Covariance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_pipeline():
    """Full optimization pipeline, callable from app.py or other scripts."""
    _, features = load_dataset()
    predictor_result = run_return_predictor()
    predicted_returns = predictor_result["latest_predictions"]

    mu, cov = build_annualized_inputs(predicted_returns, features)
    max_sharpe_weights = optimize_max_sharpe(mu, cov)
    frontier_df = compute_efficient_frontier(mu, cov)

    return {
        "mu": mu,
        "cov": cov,
        "max_sharpe_weights": max_sharpe_weights,
        "frontier": frontier_df,
    }


if __name__ == "__main__":
    result = run_pipeline()
    mu, cov = result["mu"], result["cov"]
    weights = result["max_sharpe_weights"]

    ret, vol, sharpe = portfolio_performance(weights, mu, cov)

    print("Annualized expected returns (from ML model):")
    print(mu.sort_values(ascending=False).to_string())

    print("\nMax-Sharpe portfolio weights:")
    print(weights[weights > 0.001].sort_values(ascending=False).to_string())

    print(f"\nMax-Sharpe portfolio expected annual return: {ret:.2%}")
    print(f"Max-Sharpe portfolio annual volatility:      {vol:.2%}")
    print(f"Max-Sharpe portfolio Sharpe ratio:            {sharpe:.3f}")

    plot_path = plot_efficient_frontier(result["frontier"], mu, cov, weights)
    print(f"\nSaved efficient frontier plot to {plot_path}")

    weights.to_csv(WEIGHTS_PATH, header=True)
    result["frontier"].to_csv(FRONTIER_PATH, index=False)
    print(f"Saved optimal weights to {WEIGHTS_PATH}")
    print(f"Saved frontier data to {FRONTIER_PATH}")
