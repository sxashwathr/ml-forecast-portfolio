"""
return_predictor.py
--------------------
This is the SECOND stage of the pipeline: it takes the feature table built
by data/data_loader.py and trains a machine learning model that predicts
each asset's *next-day* return from its recent behavior (volatility and
momentum).

WHY we even attempt this (and why we're honest that it's hard):
Predicting stock returns is famously difficult -- the "efficient market
hypothesis" argues that publicly known information (like recent momentum
or volatility) should already be priced in, leaving little left to
predict. We're not claiming to have found a market-beating signal. The
point of this file is to build the ML piece of the pipeline *correctly*
(no lookahead bias, honestly reported accuracy) so that later, in
backtest.py, we can honestly measure whether it actually helps -- and
report the truth either way.

WHY Ridge regression specifically (not plain linear regression or
something more complex like a random forest):
  - Our features (momentum, volatility) are noisy and only weakly related
    to next-day returns. Plain linear regression (OLS) has no protection
    against overfitting to that noise. Ridge adds an L2 penalty that
    shrinks coefficients toward zero, which is exactly the right tool
    when you suspect the true signal is weak.
  - It's simple and linear, which means we can actually explain *why* it
    predicts what it predicts (a beginner-friendly, interview-friendly
    property) -- unlike a black-box model like a random forest or neural
    net, which would be overkill for 3 features and would be much harder
    to defend as "not just overfitting."
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_dataset

# WHY we fix the seed even though Ridge's default solver is deterministic:
# scikit-learn's Ridge uses a closed-form solution by default, so it has no
# inherent randomness. But NumPy's global random state can still leak into
# other pieces of this project (e.g. if a solver falls back to an iterative
# method, or future code changes add randomness here). Setting the seed
# once, at import time, keeps this file consistent with the project-wide
# reproducibility convention rather than relying on every future edit to
# remember to do it.
np.random.seed(42)

FEATURE_COLS = ["volatility_30d", "momentum_30d", "momentum_90d"]
TARGET_COL = "target_next_return"

# WHY 0.2 (20%) held out for testing: with ~1,400 trading days per ticker,
# 20% leaves roughly a year of unseen test data per asset -- enough to
# judge performance across different market conditions, while still
# leaving 80% (~4.5 years) to train on.
TEST_FRACTION = 0.2

# WHY we search over a range of alpha (the Ridge regularization strength)
# instead of picking one number: alpha controls how hard we shrink
# coefficients toward zero. Too small and we're back to overfitting noise;
# too large and the model just predicts ~0 for everyone, ignoring the
# features entirely. Rather than guess, we let cross-validation pick the
# value that generalizes best on held-out training-period data.
ALPHA_GRID = [0.1, 1.0, 10.0, 50.0, 100.0]

PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "predicted_returns.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "model_metrics.csv")


def build_supervised_dataset(features):
    """
    Turn the feature table into a supervised-learning table by attaching a
    target column: *tomorrow's* return for each asset.

    WHY shift(-1) and not just using daily_return directly as the target:
    daily_return on a given row is the return that ALREADY happened on
    that date -- using it as both a feature-day label and training target
    would be trivial and useless. We want to predict the *next* day's
    return using today's (already-known) features, which is what actually
    happens in real trading: you have today's data, and you're trying to
    decide about tomorrow.
    """
    df = features.sort_values(["ticker", "date"]).copy()
    df[TARGET_COL] = df.groupby("ticker")["daily_return"].shift(-1)

    # The most recent day for each ticker has no "next day" yet -- its
    # target is NaN. We keep a copy of that latest row separately (it's
    # exactly what we'll feed the trained model to get a real, actionable
    # prediction for the optimizer), and drop it from the trainable set.
    latest_rows = df[df[TARGET_COL].isna()].copy()
    trainable = df.dropna(subset=[TARGET_COL]).copy()
    return trainable, latest_rows


def time_based_split(df, test_fraction=TEST_FRACTION):
    """
    Split into train/test by DATE, not randomly.

    WHY this matters (no lookahead bias): a random shuffle-split would let
    the model train on some days from 2025 and test on some days from
    2022 -- meaning it could effectively "see the future" relative to what
    it's being tested on, and a covariance/pattern that only existed
    because the model indirectly learned from later data leaking into
    earlier predictions. In real trading you never get to train on
    tomorrow to predict yesterday. A time-based split -- train on the
    earliest 80% of dates, test on the most recent 20% -- mirrors how the
    model would actually be used.

    WHY the SAME cutoff date is used across every ticker (rather than each
    ticker getting its own 80/20 split independently): portfolio decisions
    happen at a single point in time across ALL assets at once. If each
    ticker split at a different date, the "test period" for one asset
    could overlap the "train period" of another, which would make the
    later portfolio-level backtest inconsistent and harder to trust.
    """
    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * (1 - test_fraction))
    cutoff_date = unique_dates[split_idx]

    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    return train, test, cutoff_date


def select_alpha(train_df):
    """
    Pick the Ridge alpha that performs best under time-series
    cross-validation on the TRAINING data only.

    WHY TimeSeriesSplit instead of scikit-learn's default KFold: ordinary
    K-fold cross-validation shuffles rows into folds, which reintroduces
    exactly the lookahead problem the outer train/test split was designed
    to avoid. TimeSeriesSplit instead creates folds that always train on
    an earlier block of time and validate on a later block, preserving
    chronological order at every step -- including during hyperparameter
    selection, not just the final train/test split.

    WHY we never touch the held-out test set here: if we picked alpha
    based on test-set performance, the test set would stop being a
    trustworthy, unbiased measure of real-world performance -- it would
    just be another thing we tuned against. Alpha selection is done
    entirely within the training data.
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha, best_score = ALPHA_GRID[0], -np.inf

    for alpha in ALPHA_GRID:
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_train[train_idx], y_train[train_idx])
            preds = pipe.predict(X_train[val_idx])
            fold_scores.append(r2_score(y_train[val_idx], preds))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha

    return best_alpha


def train_ticker_models(trainable_df, alpha=None):
    """
    Train one Ridge model PER TICKER (rather than one pooled model across
    all assets).

    WHY per-ticker models: different assets behave differently -- NVDA's
    momentum may carry very different predictive information than SPY's.
    A single pooled model would force one set of coefficients onto every
    asset, blurring those differences. Training separately also directly
    produces exactly what the optimizer needs next: one predicted return
    per asset.

    WHY a StandardScaler is required before Ridge: Ridge's penalty
    shrinks all coefficients using the same alpha, but that's only fair if
    every feature is on a comparable scale. Momentum is often in the
    range of roughly -0.3 to 0.3, while annualized volatility is often
    0.15-0.6 -- without scaling, the penalty would shrink whichever
    feature happens to have larger raw numbers, for no principled reason.

    Returns: dict of {ticker: fitted Pipeline}, plus the alpha used.
    """
    if alpha is None:
        alpha = select_alpha(trainable_df)

    models = {}
    for ticker, group in trainable_df.groupby("ticker"):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(group[FEATURE_COLS].values, group[TARGET_COL].values)
        models[ticker] = pipe
    return models, alpha


def evaluate_models(models, train_df, test_df):
    """
    Score every ticker's model on its own held-out (future, unseen) test
    period, AND compare against a naive baseline: "predict the average
    training-period return every single day."

    WHY compare to that specific baseline: it's the simplest possible
    forecast -- pure historical average, no features, no ML. If our Ridge
    model can't beat "just guess the historical mean," the model isn't
    adding value, and we should say so plainly rather than quietly
    reporting an R^2 number without context. This is the same honesty
    standard applied later, at the portfolio level, in backtest.py.
    """
    rows = []
    for ticker, model in models.items():
        train_group = train_df[train_df["ticker"] == ticker]
        test_group = test_df[test_df["ticker"] == ticker]
        if test_group.empty:
            continue

        X_test = test_group[FEATURE_COLS].values
        y_test = test_group[TARGET_COL].values
        preds = model.predict(X_test)

        naive_baseline_pred = np.full_like(y_test, train_group[TARGET_COL].mean())

        # Directional accuracy: how often did we at least get the SIGN of
        # the next return right? For trading decisions, getting the
        # direction right is often more actionable than nailing the exact
        # magnitude, and it's a much more intuitive number for a beginner
        # to sanity-check ("would this have flagged up-days as up-days?").
        directional_accuracy = float(np.mean(np.sign(preds) == np.sign(y_test)))

        rows.append({
            "ticker": ticker,
            "test_r2": r2_score(y_test, preds),
            "test_mae": mean_absolute_error(y_test, preds),
            "baseline_mae": mean_absolute_error(y_test, naive_baseline_pred),
            "directional_accuracy": directional_accuracy,
            "n_test_days": len(test_group),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def predict_latest_returns(models, latest_rows):
    """
    Use each ticker's trained model plus its most recent feature row (the
    one row that had no "next day" to train against) to produce one
    predicted next-period return per asset.

    WHY this is exactly what feeds the optimizer: portfolio optimization
    needs a forward-looking expected return per asset (often called "mu").
    This function is where the ML model's forecast becomes that mu vector.
    """
    preds = {}
    for ticker, row in latest_rows.set_index("ticker").iterrows():
        model = models[ticker]
        X = row[FEATURE_COLS].values.reshape(1, -1)
        preds[ticker] = float(model.predict(X)[0])
    return pd.Series(preds, name="predicted_next_return").sort_index()


def run_pipeline(force_refresh=False):
    """
    Full training pipeline: load data, split by time, tune alpha, train
    per-ticker models, evaluate honestly, and predict the latest expected
    returns. Returns everything downstream code (optimizer.py, backtest.py)
    might need.
    """
    _, features = load_dataset(force_refresh=force_refresh)
    trainable, latest_rows = build_supervised_dataset(features)
    train_df, test_df, cutoff_date = time_based_split(trainable)

    models, alpha = train_ticker_models(train_df)
    metrics = evaluate_models(models, train_df, test_df)
    latest_predictions = predict_latest_returns(models, latest_rows)

    return {
        "models": models,
        "alpha": alpha,
        "cutoff_date": cutoff_date,
        "metrics": metrics,
        "latest_predictions": latest_predictions,
    }


if __name__ == "__main__":
    result = run_pipeline()

    print(f"Selected Ridge alpha (via time-series CV): {result['alpha']}")
    print(f"Train/test split cutoff date: {pd.Timestamp(result['cutoff_date']).date()}")
    print(f"\nPer-ticker test-period performance:\n{result['metrics'].to_string(index=False)}")

    avg_r2 = result["metrics"]["test_r2"].mean()
    avg_dir_acc = result["metrics"]["directional_accuracy"].mean()
    beats_baseline = (result["metrics"]["test_mae"] < result["metrics"]["baseline_mae"]).sum()
    total = len(result["metrics"])
    print(f"\nAverage test R^2 across tickers: {avg_r2:.4f}")
    print(f"Average directional accuracy: {avg_dir_acc:.2%}")
    print(f"Beat the naive historical-mean baseline (lower MAE) on {beats_baseline}/{total} tickers")
    print(
        "\nNote: low/negative R^2 here is EXPECTED and honestly reported -- "
        "next-day stock returns are extremely hard to predict from momentum "
        "and volatility alone. This is not a bug; see backtest.py for whether "
        "this still helps at the portfolio level."
    )

    print(f"\nLatest predicted next-period returns (feeds the optimizer):\n{result['latest_predictions']}")

    result["latest_predictions"].to_csv(PREDICTIONS_PATH, header=True)
    result["metrics"].to_csv(METRICS_PATH, index=False)
    print(f"\nSaved predictions to {PREDICTIONS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
