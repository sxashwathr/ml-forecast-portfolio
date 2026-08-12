"""
data_loader.py
---------------
This is the FIRST stage of the pipeline: it turns raw stock prices into a
clean, feature-rich table that the rest of the project (the ML model, the
optimizer, the simulator, the backtester) can all build on top of.

WHY this file exists at all:
Every downstream step -- predicting returns, optimizing a portfolio,
simulating risk -- needs the same two raw ingredients: (1) a history of
prices, and (2) a set of numeric "features" describing each asset's recent
behavior. If every other file pulled its own data and computed its own
features, we'd risk subtle inconsistencies (e.g. one file using 20-day
volatility and another using 30-day). Centralizing it here means every
other module sees the exact same numbers.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# WHY these specific tickers:
# The user wants a portfolio themed around robotics, automation, and
# deep-tech, plus a couple of broad market ETFs as a "sanity check" balance.
# Broad ETFs matter because they represent the diversified market baseline --
# if our themed picks can't do better (risk-adjusted) than just owning the
# whole market, that's an important, honest finding, not a bug.
#
# All of these have listed for well over 5 years, which matters because we
# need a long, continuous price history to compute rolling features and to
# have enough data for a meaningful train/test split later.
# ---------------------------------------------------------------------------
ROBOTICS_AUTOMATION_TICKERS = [
    "ISRG",  # Intuitive Surgical -- surgical robotics
    "ROK",   # Rockwell Automation -- industrial automation/controls
    "PH",    # Parker Hannifin -- motion control & industrial automation systems
    "TER",   # Teradyne -- owns Universal Robots/MiR, automated test equipment
    "CGNX",  # Cognex -- machine vision systems used in robotics/automation
    "NVDA",  # Nvidia -- compute/AI chips underpinning modern robotics & deep-tech
    "EMR",   # Emerson Electric -- industrial automation
    "HON",   # Honeywell -- industrial automation & deep-tech conglomerate
]
BROAD_ETFS = [
    "SPY",   # S&P 500 -- broad US equity market baseline
    "QQQ",   # Nasdaq-100 -- broad, tech-tilted market baseline
]
TICKERS = ROBOTICS_AUTOMATION_TICKERS + BROAD_ETFS

# WHY 6 years of raw history instead of exactly 5:
# We need >= 5 years of *usable* data. But our rolling features (30-day
# volatility, 90-day momentum) each "eat" the first N days of history
# producing NaNs while there isn't enough of a window yet. Pulling an extra
# year of buffer means that after we drop those NaN warm-up rows, we still
# comfortably have 5+ years of usable, feature-complete data.
LOOKBACK_PERIOD = "6y"

# WHY a fixed rolling window size instead of something dynamic:
# 30 trading days is roughly one calendar month -- a common, interpretable
# convention in finance for "recent" volatility. Keeping it fixed (rather
# than tuned) keeps the project transparent: we're not secretly overfitting
# our feature definitions to make the model look better.
VOLATILITY_WINDOW = 30
MOMENTUM_WINDOWS = [30, 90]  # ~1 month and ~1 quarter momentum

RAW_PRICES_PATH = os.path.join(os.path.dirname(__file__), "raw_prices.csv")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features.csv")


def download_price_data(tickers=TICKERS, period=LOOKBACK_PERIOD):
    """
    Pull daily price history for all tickers from Yahoo Finance.

    WHY we use `auto_adjust=True` (yfinance's default): raw closing prices
    jump around on stock splits and dividend payments in ways that have
    nothing to do with actual investment performance. "Adjusted close"
    corrects for that, so a return computed from it reflects what an
    investor who held the asset actually experienced.

    Returns a wide DataFrame indexed by Date, with one "Close" column per
    ticker.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex column structure like (Price, Ticker)
    # when given multiple tickers. We only need the Close price for every
    # downstream calculation (returns, volatility, momentum all derive from
    # it), so we select just that slice and flatten it into a simple
    # Date x Ticker table -- much easier for everything else in this
    # project to work with.
    prices = raw["Close"].copy()
    prices = prices.dropna(how="all")  # drop rows where nothing traded (e.g. holidays that slipped through)
    return prices


def compute_features(prices):
    """
    Turn a wide Date x Ticker price table into a tidy, "long" feature table:
    one row per (Date, Ticker) with columns for the price, the daily return,
    rolling volatility, and momentum.

    WHY "long" (tidy) format instead of keeping everything wide: step 2
    (the ML model) needs to treat every (asset, day) pair as one training
    example, with the same feature columns across every asset. A long
    format is the natural, standard shape for that -- and it's what
    pandas groupby/rolling operations are built around.

    WHY these specific features:
      - daily_return: the thing we ultimately want to predict/optimize for.
      - volatility_30d: recent risk. A model or optimizer that only looks
        at returns and ignores how volatile an asset has been is flying
        half-blind -- volatility is a core input to Sharpe ratio and to
        the covariance matrix used in optimization.
      - momentum_30d / momentum_90d: cumulative return over the trailing
        window. Momentum (the tendency for recent winners to keep
        performing well over medium horizons) is one of the most
        well-documented, persistent patterns in equity markets, which
        makes it a reasonable, non-exotic feature to hand a beginner-level
        model.
    """
    long_rows = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        df = pd.DataFrame(index=series.index)
        df.index.name = "date"  # yfinance names the index "Date"; pin it to a known name so reset_index() below is predictable
        df["ticker"] = ticker
        df["close"] = series

        # Daily simple return: (today's price / yesterday's price) - 1.
        # WHY simple (not log) returns: they're easier for a beginner to
        # reason about ("+2% today") and they aggregate correctly with
        # portfolio weights (a weighted average of simple returns equals
        # the portfolio's simple return -- that's not true for log returns).
        df["daily_return"] = series.pct_change()

        # Rolling 30-day volatility of daily returns, annualized.
        # WHY annualize (multiply by sqrt(252)): a "daily std dev" number
        # like 0.015 is hard to build intuition for. Multiplying by the
        # square root of the number of trading days in a year (~252)
        # converts it into a "per year" volatility figure that lines up
        # with how volatility is normally quoted (and with how we'll later
        # compute the Sharpe ratio).
        df["volatility_30d"] = (
            df["daily_return"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)
        )

        # Momentum: cumulative return over the trailing N days. This uses
        # price directly (price_today / price_N_days_ago - 1) rather than
        # summing daily returns, because compounding daily returns and
        # taking a simple cumulative price ratio are equivalent -- and the
        # price-ratio version is more numerically direct and easier to
        # explain.
        for window in MOMENTUM_WINDOWS:
            df[f"momentum_{window}d"] = series / series.shift(window) - 1.0

        long_rows.append(df)

    features = pd.concat(long_rows).reset_index()

    # WHY we drop rows with any NaN feature: the longest warm-up period is
    # the 90-day momentum window, so the first ~90 trading days of each
    # ticker's history can't have every feature computed yet. Rather than
    # filling those with fabricated placeholder values (which would quietly
    # inject fake signal), we drop them. We pulled 6 years of raw data
    # specifically so that after this drop, 5+ years of complete data remain.
    features = features.dropna().reset_index(drop=True)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    return features


def load_dataset(force_refresh=False):
    """
    Main entry point for the rest of the project: returns (prices, features).

    WHY we cache to CSV: hitting Yahoo Finance's API on every single run of
    every downstream script (model training, optimization, simulation,
    backtesting, the dashboard) is slow, network-dependent, and -- since
    prices update in real time -- would make results change between runs,
    undermining the reproducibility the whole project is built around.
    Caching once to disk means every other module reads the *same* dataset
    until we deliberately refresh it.
    """
    if not force_refresh and os.path.exists(RAW_PRICES_PATH) and os.path.exists(FEATURES_PATH):
        prices = pd.read_csv(RAW_PRICES_PATH, index_col=0, parse_dates=True)
        features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
        return prices, features

    prices = download_price_data()
    features = compute_features(prices)

    prices.to_csv(RAW_PRICES_PATH)
    features.to_csv(FEATURES_PATH, index=False)
    return prices, features


if __name__ == "__main__":
    # Running this file directly downloads fresh data and prints a summary,
    # which is the fastest way to sanity-check that everything worked before
    # any other module depends on it.
    prices, features = load_dataset(force_refresh=True)

    print(f"Downloaded prices for {prices.shape[1]} tickers, {prices.shape[0]} trading days")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"\nFeature table: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Columns: {list(features.columns)}")
    print(f"\nSample rows:\n{features.head()}")
    print(f"\nPer-ticker row counts:\n{features['ticker'].value_counts()}")
