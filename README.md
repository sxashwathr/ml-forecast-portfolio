# ML Forecast Portfolio

A portfolio optimization project. A Ridge regression model
forecasts next-period returns for a basket of stocks, forecasts feed
into an SLSQP optimizer to build a max-Sharpe portfolio, a Monte Carlo
simulation stress-tests the downside, and a backtest checks
the significance of the added ML layer.

## What's in the pipeline

- **Data**: 5+ years of daily prices for 10 tickers, 8 robotics/automation
  names (`ISRG`, `ROK`, `PH`, `TER`, `CGNX`, `NVDA`, `EMR`, `HON`) plus SPY
  and QQQ as a baseline.
- **Forecasting**: Ridge regression predicts next-day returns per stock
  using rolling volatility and momentum, split strictly by date so nothing
  leaks from the future.
- **Optimization**: SLSQP finds the max-Sharpe portfolio and traces the
  efficient frontier.
- **Risk simulation**: 10,000 bootstrapped paths give VaR and Expected
  Shortfall for the chosen portfolio.
- **Backtest**: walk-forward, monthly rebalancing, compares equal-weight
  vs. naive-mean optimization vs. ML-based optimization.
- **Dashboard**: Streamlit app with a risk slider tied to the frontier.

Random seed is fixed at 42 everywhere, so test results can be replicated.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python data/data_loader.py
python models/return_predictor.py
python optimization/optimizer.py
python simulation/monte_carlo.py
python backtest/backtest.py

streamlit run app.py
```

## Did the ML actually help?

The added ML layer helped, but only to a short extent.

| Strategy | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Equal-weight | 188.6% | 1.22 | -28.4% |
| Naive historical mean | 660.7% | **1.58** | -28.7% |
| ML-predicted | 589.5% | 1.56 | -26.9% |

The ML version beat doing nothing (equal-weight), but it didn't beat just
using the plain historical average return apart from a short stretch in the simulation (Dec. 2024 - May 2025). This makes sense because the model's R²
came out to -0.009, basically zero, and it's directional accuracy was 50.8%: essentially a
coin flip. Feeding weak signals into optimizers doesn't make it a
strong signal, but rather makes the optimizer  bet on noise.
This is what happened, since the ML version put 83% of the portfolio
into HON based on a return forecast that had no real predictive power.

A few honest limitations: no transaction costs anywhere in the backtest,
a pretty small feature set (just momentum and volatility), the Monte
Carlo bootstrap samples days independently so it misses how bad days
tend to cluster together, and 10 tickers is a small, thematic universe,
not something you'd actually diversify a real portfolio with.

The point of this project was never to prove that "ML beats simple methods." It
was made to test that question honestly instead of assuming an
answer, and report what came out, even when not entirely significant.
