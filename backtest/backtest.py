"""
backtest.py
-----------
This is the FIFTH stage of the pipeline, and arguably the most important:
it's the moment where we actually find out whether any of this was worth
building.

WHY a WALK-FORWARD backtest specifically (not just "run the optimizer once
on all historical data and see how it would have done"):
Everything computed so far (the ML predictions, the optimal weights) used
the FULL dataset, including dates that are now in the "past" relative to
each other but were fit all at once. That's unrealistic -- on any given day
in 2022, a real investor would only have had data up through that day, not
data through 2026. A walk-forward backtest fixes this by repeatedly: (1)
looking only at a trailing window of history, (2) making a decision (asset
weights) using ONLY that window, (3) holding those weights for the next
short period and recording what actually happened, (4) rolling the window
forward and repeating. This is the closest a backtest can get to honestly
simulating "what would have happened if I'd actually traded this."

WHY we compare THREE strategies side by side, using the exact same
covariance matrix and rebalancing schedule for all three:
  (a) Equal-weight: the simplest possible baseline -- 1/N in every asset,
      no optimization or prediction at all. If nothing else can beat this,
      that's an important, humbling result.
  (b) Naive-mean-optimized: same SLSQP optimizer as the ML version, but
      fed a naive expected return estimate (just the historical average
      return over the trailing window) instead of an ML prediction. This
      isolates ONE variable: does the ML model's return forecast add value
      over "just use the recent average," given that everything else
      (optimizer, constraints, covariance, rebalance timing) is identical?
  (c) ML-predicted-optimized: the full pipeline -- Ridge-predicted returns
      feeding the same optimizer.
Using the same covariance matrix and schedule for (b) and (c) means any
performance difference between them is attributable to the RETURN
ESTIMATE, not to some other confound -- which is the fair, honest way to
test whether the ML model specifically is pulling its weight.
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
from models.return_predictor import build_supervised_dataset, time_based_split, train_ticker_models, predict_latest_returns
from optimization.optimizer import optimize_max_sharpe, RISK_FREE_RATE_ANNUAL, TRADING_DAYS_PER_YEAR

# WHY 504 trading days (~2 years) as the rolling estimation window: this
# needs to be long enough to (a) give the Ridge model enough rows to train
# on (recall it further splits this window 80/20 internally) and (b) give
# a reasonably stable covariance/mean estimate, while still being short
# enough to reflect a "recent regime" rather than averaging in stale
# behavior from years ago that may no longer be representative.
ROLLING_WINDOW_DAYS = 504

# WHY rebalance every 21 trading days (~1 month): monthly rebalancing is a
# common, realistic cadence for a retail-scale portfolio -- frequent enough
# to react to changing conditions, infrequent enough to keep the number of
# walk-forward steps (and, in a real setting, transaction costs) reasonable.
REBALANCE_EVERY_DAYS = 21

COMPARISON_PLOT_PATH = os.path.join(os.path.dirname(__file__), "backtest_comparison.png")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
CUMULATIVE_RETURNS_PATH = os.path.join(os.path.dirname(__file__), "cumulative_returns.csv")


def get_aligned_wide_returns(features):
    """
    Same idea as in optimizer.py/monte_carlo.py: pivot the long feature
    table into a Date x Ticker table of daily returns, keeping only dates
    where every asset has data. This is the ground truth we replay forward
    through the backtest -- it's not re-estimated at each step, only used
    to (a) build each step's trailing-window inputs and (b) score what a
    given set of weights would have actually earned.
    """
    return features.pivot(index="date", columns="ticker", values="daily_return").dropna()


def estimate_naive_mean_and_cov(window_returns_wide):
    """
    The "naive" strategy's return estimate: simply the average daily return
    over the trailing window, annualized. No model, no features -- just
    "assume the recent past repeats." This -- along with the identical
    covariance estimate used by the ML strategy -- is the deliberately
    simple baseline the ML approach has to beat to justify its complexity.
    """
    mu = window_returns_wide.mean() * TRADING_DAYS_PER_YEAR
    cov = window_returns_wide.cov() * TRADING_DAYS_PER_YEAR
    return mu, cov


def estimate_ml_mu(window_features, tickers):
    """
    The ML strategy's return estimate for this window: retrain per-ticker
    Ridge models using ONLY data inside the current trailing window (no
    peeking beyond it), then predict from the window's most recent feature
    row -- exactly the reusable logic from models/return_predictor.py,
    just applied to a rolling slice instead of the full dataset.

    WHY retrain from scratch at every step instead of training once: in a
    walk-forward setup, "today's" trailing window is different at every
    step -- retraining each time is what makes each prediction genuinely
    only informed by data available as of that point in time.
    """
    trainable, latest_rows = build_supervised_dataset(window_features)
    train_df, _test_df, _cutoff = time_based_split(trainable)
    models, _alpha = train_ticker_models(train_df)
    latest_preds = predict_latest_returns(models, latest_rows)
    mu_daily = latest_preds.reindex(tickers)
    return mu_daily * TRADING_DAYS_PER_YEAR


def run_walk_forward_backtest():
    """
    The core walk-forward loop. At each rebalance step:
      1. Look at the trailing ROLLING_WINDOW_DAYS of history (and nothing
         after it).
      2. Compute weights for all three strategies from that window alone.
      3. Apply those (now-fixed) weights to the NEXT REBALANCE_EVERY_DAYS
         of ACTUAL realized returns -- data the strategies did not see
         when the weights were chosen.
      4. Record those realized returns, then roll the window forward and
         repeat.

    Returns a dict of {strategy_name: full concatenated out-of-sample daily
    return series}, spanning the entire backtest period.
    """
    _, features = load_dataset()
    wide_returns = get_aligned_wide_returns(features)
    tickers = wide_returns.columns
    dates = wide_returns.index
    n_days = len(dates)

    strategy_return_chunks = {"equal_weight": [], "naive_mean": [], "ml_predicted": []}
    weight_history = []

    step_start = ROLLING_WINDOW_DAYS
    while step_start < n_days:
        step_end = min(step_start + REBALANCE_EVERY_DAYS, n_days)
        window_dates = dates[step_start - ROLLING_WINDOW_DAYS: step_start]
        holding_dates = dates[step_start: step_end]
        if len(holding_dates) == 0:
            break

        window_returns_wide = wide_returns.loc[window_dates]
        window_features = features[features["date"].isin(window_dates)]

        mu_naive, cov = estimate_naive_mean_and_cov(window_returns_wide)
        mu_ml = estimate_ml_mu(window_features, tickers)

        w_equal = pd.Series(1.0 / len(tickers), index=tickers)
        w_naive = optimize_max_sharpe(mu_naive, cov)
        w_ml = optimize_max_sharpe(mu_ml, cov)

        holding_returns = wide_returns.loc[holding_dates]
        strategy_return_chunks["equal_weight"].append(holding_returns @ w_equal)
        strategy_return_chunks["naive_mean"].append(holding_returns @ w_naive)
        strategy_return_chunks["ml_predicted"].append(holding_returns @ w_ml)

        weight_history.append({
            "rebalance_date": window_dates[-1],
            **{f"naive_{t}": w_naive[t] for t in tickers},
            **{f"ml_{t}": w_ml[t] for t in tickers},
        })

        step_start = step_end

    strategy_returns = {name: pd.concat(chunks) for name, chunks in strategy_return_chunks.items()}
    weight_history_df = pd.DataFrame(weight_history)
    return strategy_returns, weight_history_df


def compute_performance_metrics(returns_series, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    """
    Summarize a daily out-of-sample return series into the three headline
    numbers requested: cumulative return, Sharpe ratio, and max drawdown.

    WHY max drawdown matters alongside Sharpe: Sharpe ratio is an AVERAGE
    risk-adjusted measure -- it can look fine even for a strategy that had
    one terrifying 40% peak-to-trough decline along the way, as long as it
    recovered. Max drawdown directly answers "what's the worst peak-to-
    valley loss an investor actually would have sat through," which is
    often a more visceral, decision-relevant number than Sharpe alone.
    """
    cumulative_return = float((1 + returns_series).prod() - 1)

    n_days = len(returns_series)
    annualized_return = float((1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)

    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess_returns = returns_series - daily_rf
    sharpe_ratio = (
        float(excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess_returns.std() > 0 else float("nan")
    )

    wealth_index = (1 + returns_series).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = wealth_index / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "n_out_of_sample_days": n_days,
    }


def plot_comparison(strategy_returns, save_path=COMPARISON_PLOT_PATH):
    """
    Plot cumulative growth of $1 for all three strategies on the same
    axes, so the reader can SEE how they diverged over time, not just
    compare summary numbers at the end.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = {
        "equal_weight": "Equal-Weight (baseline)",
        "naive_mean": "Optimized -- Naive Historical Mean",
        "ml_predicted": "Optimized -- ML-Predicted Returns",
    }
    for name, returns_series in strategy_returns.items():
        wealth_index = (1 + returns_series).cumprod()
        ax.plot(wealth_index.index, wealth_index.values, linewidth=2, label=labels[name])

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (out-of-sample, walk-forward)")
    ax.set_title("Walk-Forward Backtest: Equal-Weight vs. Naive-Mean vs. ML-Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_pipeline():
    strategy_returns, weight_history_df = run_walk_forward_backtest()
    metrics = {name: compute_performance_metrics(series) for name, series in strategy_returns.items()}
    metrics_df = pd.DataFrame(metrics).T
    return {
        "strategy_returns": strategy_returns,
        "weight_history": weight_history_df,
        "metrics": metrics_df,
    }


if __name__ == "__main__":
    result = run_pipeline()
    metrics_df = result["metrics"]

    print(f"Walk-forward backtest: {ROLLING_WINDOW_DAYS}-day rolling window, "
          f"rebalanced every {REBALANCE_EVERY_DAYS} trading days")
    print(f"Out-of-sample period length: {int(metrics_df['n_out_of_sample_days'].iloc[0])} trading days\n")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # WHY this explicit, unhedged verdict block: the whole point of the
    # backtest is to answer a yes/no question honestly. Burying the answer
    # in a table and letting the reader guess would defeat that purpose.
    ml_sharpe = metrics_df.loc["ml_predicted", "sharpe_ratio"]
    naive_sharpe = metrics_df.loc["naive_mean", "sharpe_ratio"]
    equal_sharpe = metrics_df.loc["equal_weight", "sharpe_ratio"]

    print("\n--- Honest verdict ---")
    if ml_sharpe > naive_sharpe and ml_sharpe > equal_sharpe:
        print(f"The ML-predicted strategy achieved the HIGHEST Sharpe ratio ({ml_sharpe:.3f}) "
              f"of the three, ahead of naive-mean ({naive_sharpe:.3f}) and equal-weight ({equal_sharpe:.3f}).")
    elif ml_sharpe > equal_sharpe:
        print(f"The ML-predicted strategy beat equal-weight ({ml_sharpe:.3f} vs {equal_sharpe:.3f}) "
              f"but did NOT beat the naive historical-mean strategy ({naive_sharpe:.3f}). "
              f"This suggests the ML model's forecasts were not more useful than simply using "
              f"recent historical averages, at this rebalancing frequency and window.")
    else:
        print(f"The ML-predicted strategy did NOT outperform equal-weight ({ml_sharpe:.3f} vs {equal_sharpe:.3f} Sharpe). "
              f"Given the near-zero R^2 the model showed in models/return_predictor.py, this is not surprising: "
              f"weak, noisy return predictions -- once fed through a Sharpe-maximizing optimizer that is highly "
              f"sensitive to input error -- do not reliably translate into better real-world portfolio performance. "
              f"This is an honest, expected finding, not a failed experiment.")

    plot_path = plot_comparison(result["strategy_returns"])
    print(f"\nSaved comparison chart to {plot_path}")

    metrics_df.to_csv(RESULTS_PATH)
    pd.DataFrame(result["strategy_returns"]).to_csv(CUMULATIVE_RETURNS_PATH)
    print(f"Saved metrics to {RESULTS_PATH}")
    print(f"Saved daily out-of-sample returns to {CUMULATIVE_RETURNS_PATH}")
