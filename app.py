"""
app.py
------
This is the SIXTH and final pipeline stage: a Streamlit dashboard that ties
data_loader.py, return_predictor.py, optimizer.py, monte_carlo.py, and
backtest.py together into one interactive view.

WHY a dashboard on top of the scripts we already have (rather than just
running each script's __main__ block): the individual scripts are great
for understanding and verifying each pipeline stage in isolation, but a
real user wants to explore ONE coherent question interactively -- "given
my risk tolerance, what portfolio would this suggest, and what does its
risk/return profile actually look like?" -- without re-reading five
separate terminal printouts.

WHY we cache the heavy pipeline stages with st.cache_resource: Streamlit
reruns the ENTIRE script top-to-bottom every time a user touches a widget
(like the risk tolerance slider). Without caching, moving the slider would
re-download price data, retrain the Ridge models, and re-run the 43-step
walk-forward backtest (~6 seconds) on every single interaction, which would
make the app feel broken. Caching means the expensive pipeline work runs
ONCE, and slider interactions only recompute the cheap, genuinely
slider-dependent parts (picking a frontier point, re-simulating Monte
Carlo for that specific portfolio).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization.optimizer import run_pipeline as run_optimizer, RISK_FREE_RATE_ANNUAL
from simulation.monte_carlo import (
    compute_portfolio_historical_returns,
    simulate_paths,
    compute_risk_metrics,
    STARTING_VALUE,
)
from backtest.backtest import run_pipeline as run_backtest
from data.data_loader import load_dataset

st.set_page_config(page_title="ML Forecast Portfolio", layout="wide")


# ---------------------------------------------------------------------------
# WHY st.cache_resource specifically (not st.cache_data): the optimizer's
# and predictor's return values include fitted scikit-learn Pipeline
# objects, which st.cache_data would try (and struggle) to hash/serialize
# for its cache key. st.cache_resource is designed exactly for caching
# "expensive-to-build Python objects" like models and keeps a single
# shared instance instead, which is both correct here and faster.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and running optimizer (first load only)...")
def get_optimizer_result():
    return run_optimizer()


@st.cache_resource(show_spinner="Loading feature data...")
def get_features():
    _, features = load_dataset()
    return features


@st.cache_resource(show_spinner="Running walk-forward backtest (this only happens once)...")
def get_backtest_result():
    return run_backtest()


def select_portfolio_for_risk_tolerance(frontier_df, risk_tolerance, mu_index):
    """
    Map a 1-10 "risk tolerance" slider value onto a specific point on the
    efficient frontier.

    WHY this is the right way to make "risk tolerance" mean something
    concrete: a vague slider labeled "risk tolerance" that doesn't actually
    change the recommended portfolio would be decorative, not useful. Here,
    1 (most conservative) maps to the LOWEST-volatility portfolio on the
    frontier, and 10 (most aggressive) maps to the HIGHEST-volatility
    portfolio on the frontier -- with everything in between linearly
    interpolated. This mirrors how real robo-advisors turn a simple
    questionnaire answer into an actual asset allocation.
    """
    vol_min, vol_max = frontier_df["volatility"].min(), frontier_df["volatility"].max()
    target_vol = vol_min + (risk_tolerance - 1) / 9.0 * (vol_max - vol_min)

    closest_idx = (frontier_df["volatility"] - target_vol).abs().idxmin()
    row = frontier_df.loc[closest_idx]

    weight_cols = [c for c in frontier_df.columns if c.startswith("weight_")]
    weights = pd.Series(
        {col.replace("weight_", ""): row[col] for col in weight_cols},
        name="weight",
    ).reindex(mu_index)
    return weights, row["return"], row["volatility"], row["sharpe"]


# ---------------------------------------------------------------------------
# Sidebar: the one user-facing control in this whole app.
# ---------------------------------------------------------------------------
st.sidebar.title("Risk Tolerance")
st.sidebar.write(
    "Moves your recommended portfolio along the efficient frontier -- "
    "1 targets the lowest available risk, 10 targets the highest."
)
risk_tolerance = st.sidebar.slider(
    "Risk tolerance (1 = Conservative, 10 = Aggressive)", min_value=1, max_value=10, value=5
)
st.sidebar.caption(
    f"Assumed risk-free rate: {RISK_FREE_RATE_ANNUAL:.1%}/year "
    "(used in every Sharpe ratio shown on this page)."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("ML Forecast Portfolio")
st.write(
    "An end-to-end pipeline: Ridge regression forecasts each asset's next-period "
    "return, an SLSQP optimizer builds a max-Sharpe portfolio from those forecasts, "
    "a Monte Carlo simulation stress-tests its risk, and a walk-forward backtest "
    "honestly checks whether the ML forecasts actually helped."
)

optimizer_result = get_optimizer_result()
mu, cov = optimizer_result["mu"], optimizer_result["cov"]
frontier_df = optimizer_result["frontier"]

weights, sel_return, sel_volatility, sel_sharpe = select_portfolio_for_risk_tolerance(
    frontier_df, risk_tolerance, mu.index
)

# ---------------------------------------------------------------------------
# Section 1: Efficient Frontier
# ---------------------------------------------------------------------------
st.header("1. Efficient Frontier")
st.write(
    "Each dot is a single asset. The blue curve is the set of portfolios offering "
    "the best possible return for a given level of risk. The diamond marks the "
    "portfolio matching your current risk tolerance; the star is the max-Sharpe portfolio."
)

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"], y=frontier_df["return"],
        mode="lines", name="Efficient Frontier", line=dict(color="royalblue", width=3),
    ))
    for ticker in mu.index:
        fig.add_trace(go.Scatter(
            x=[np.sqrt(cov.loc[ticker, ticker])], y=[mu[ticker]],
            mode="markers+text", text=[ticker], textposition="top center",
            marker=dict(size=10), name=ticker, showlegend=False,
        ))
    max_sharpe_weights = optimizer_result["max_sharpe_weights"]
    ms_return = float(max_sharpe_weights @ mu)
    ms_vol = float(np.sqrt(max_sharpe_weights @ cov.values @ max_sharpe_weights))
    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_return], mode="markers", name="Max-Sharpe Portfolio",
        marker=dict(symbol="star", size=18, color="gold", line=dict(color="black", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=[sel_volatility], y=[sel_return], mode="markers", name="Your Risk Tolerance",
        marker=dict(symbol="diamond", size=16, color="crimson", line=dict(color="black", width=1)),
    ))
    fig.update_layout(
        xaxis_title="Annualized Volatility (Risk)", yaxis_title="Annualized Expected Return",
        height=520, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.metric("Expected Annual Return", f"{sel_return:.2%}")
    st.metric("Annual Volatility", f"{sel_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{sel_sharpe:.2f}")
    st.write("**Portfolio weights:**")
    display_weights = weights[weights > 0.001].sort_values(ascending=False)
    st.dataframe(display_weights.map(lambda w: f"{w:.1%}"), width="stretch")

# ---------------------------------------------------------------------------
# Section 2: Monte Carlo Fan Chart
# ---------------------------------------------------------------------------
st.header("2. Monte Carlo Risk Simulation")
st.write(
    "10,000 bootstrapped 1-year paths for the portfolio matching your selected risk "
    "tolerance, built by resampling that portfolio's own historical daily returns."
)

features = get_features()
# WHY re-simulate on every slider move instead of caching this step too:
# unlike the optimizer/backtest, this depends directly on the risk-tolerance
# slider (different weights -> different historical portfolio-return series
# to bootstrap from). But it's fully vectorized NumPy work on ~10,000
# samples, so it re-runs in a fraction of a second -- fast enough that
# caching it would add complexity for no real speed benefit.
historical_returns = compute_portfolio_historical_returns(features, weights.fillna(0.0))
paths = simulate_paths(historical_returns)
mc_metrics = compute_risk_metrics(paths)

mc_col1, mc_col2 = st.columns([2, 1])

with mc_col1:
    days = np.arange(paths.shape[1])
    p5, p25, p50, p75, p95 = (np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95))

    fan_fig = go.Figure()
    fan_fig.add_trace(go.Scatter(x=days, y=p95, line=dict(width=0), showlegend=False))
    fan_fig.add_trace(go.Scatter(x=days, y=p5, fill="tonexty", fillcolor="rgba(70,130,180,0.2)",
                                  line=dict(width=0), name="5th-95th percentile"))
    fan_fig.add_trace(go.Scatter(x=days, y=p75, line=dict(width=0), showlegend=False))
    fan_fig.add_trace(go.Scatter(x=days, y=p25, fill="tonexty", fillcolor="rgba(70,130,180,0.4)",
                                  line=dict(width=0), name="25th-75th percentile"))
    fan_fig.add_trace(go.Scatter(x=days, y=p50, line=dict(color="navy", width=2), name="Median path"))
    fan_fig.add_hline(y=STARTING_VALUE, line_dash="dash", line_color="gray")
    fan_fig.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Simulated Portfolio Value ($)", height=480)
    st.plotly_chart(fan_fig, width="stretch")

with mc_col2:
    st.metric("Median 1-Year Value", f"${mc_metrics['median_terminal_value']:,.0f}")
    st.metric("Value at Risk (95%)", f"{mc_metrics['VaR_95']:.1%}")
    st.metric("Expected Shortfall (95%)", f"{mc_metrics['expected_shortfall_95']:.1%}")
    st.metric("Probability of Loss (1yr)", f"{mc_metrics['probability_of_loss_1yr']:.1%}")

# ---------------------------------------------------------------------------
# Section 3: Backtest Comparison
# ---------------------------------------------------------------------------
st.header("3. Walk-Forward Backtest: Does the ML Model Actually Help?")
st.write(
    "This section is NOT affected by the risk-tolerance slider -- it compares three "
    "fixed strategies (equal-weight, optimized on naive historical-mean returns, and "
    "optimized on ML-predicted returns) using the same walk-forward methodology from "
    "backtest.py, so the comparison stays apples-to-apples regardless of your risk setting."
)

backtest_result = get_backtest_result()
strategy_returns = backtest_result["strategy_returns"]
metrics_df = backtest_result["metrics"]

bt_col1, bt_col2 = st.columns([2, 1])

with bt_col1:
    labels = {
        "equal_weight": "Equal-Weight (baseline)",
        "naive_mean": "Optimized -- Naive Historical Mean",
        "ml_predicted": "Optimized -- ML-Predicted Returns",
    }
    bt_fig = go.Figure()
    for name, series in strategy_returns.items():
        wealth_index = (1 + series).cumprod()
        bt_fig.add_trace(go.Scatter(x=wealth_index.index, y=wealth_index.values, name=labels[name], line=dict(width=2)))
    bt_fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    bt_fig.update_layout(xaxis_title="Date", yaxis_title="Growth of $1 (out-of-sample)", height=480,
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(bt_fig, width="stretch")

with bt_col2:
    st.write("**Performance summary:**")
    display_metrics = metrics_df[["cumulative_return", "sharpe_ratio", "max_drawdown"]].rename(
        index=labels,
        columns={"cumulative_return": "Cumulative Return", "sharpe_ratio": "Sharpe Ratio", "max_drawdown": "Max Drawdown"},
    )
    st.dataframe(
        display_metrics.style.format({"Cumulative Return": "{:.1%}", "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.1%}"}),
        width="stretch",
    )

    ml_sharpe = metrics_df.loc["ml_predicted", "sharpe_ratio"]
    naive_sharpe = metrics_df.loc["naive_mean", "sharpe_ratio"]
    equal_sharpe = metrics_df.loc["equal_weight", "sharpe_ratio"]
    if ml_sharpe > naive_sharpe and ml_sharpe > equal_sharpe:
        st.success("The ML strategy achieved the highest Sharpe ratio of the three.")
    elif ml_sharpe > equal_sharpe:
        st.warning(
            "The ML strategy beat equal-weight but NOT the naive historical-mean "
            "strategy -- the ML forecasts didn't add value over a simple average here."
        )
    else:
        st.error(
            "The ML strategy did not outperform equal-weight. Given the near-zero R^2 "
            "seen in return_predictor.py, this is an honest, expected result -- not "
            "a bug in the pipeline."
        )

st.divider()
st.caption(
    "Educational project. Not investment advice. All results use a fixed random seed "
    "(42) wherever randomness is involved, so they're reproducible on any machine."
)
