# EXPLAINED: How This Project Works, Start to Finish

This document walks through the entire ML Forecast Portfolio project in
plain English — assuming you know basic Python, but not finance or ML —
and is written so you can read it once and then defend every design
decision in an interview. It follows the same order the code runs in.

---

## 1. The one-sentence pitch

*"I built a pipeline that predicts stock returns with machine learning,
feeds those predictions into a portfolio optimizer, stress-tests the
result with a Monte Carlo simulation, and then honestly backtests whether
the ML model actually made the portfolio better than much simpler
alternatives — and it's a nuanced 'partially.'"*

That last clause is the most important part of the pitch. Anyone can build
a pipeline that produces a portfolio. The harder, more valuable thing —
and the thing this project is actually about — is honestly measuring
whether the fancy part (the ML) earned its complexity.

---

## 2. The big picture: how data flows through the project

```
Yahoo Finance prices
        │
        ▼
data_loader.py       →  daily prices + volatility/momentum features
        │
        ▼
return_predictor.py  →  per-asset Ridge model  →  predicted next-day returns
        │
        ▼
optimizer.py          →  SLSQP max-Sharpe optimizer  →  portfolio weights
        │                          + efficient frontier
        ▼
monte_carlo.py        →  10,000 bootstrapped 1-year paths  →  VaR, Expected Shortfall
        │
        ▼
backtest.py            →  walk-forward test: does the ML step actually help?
        │
        ▼
app.py                 →  Streamlit dashboard tying it all together
```

Each stage's output is cached to disk (CSV/PNG) specifically so the next
stage — and the dashboard — never has to silently redo expensive or
network-dependent work.

---

## 3. Stage 1 — `data_loader.py`: turning raw prices into features

**What it does:** Downloads ~6 years of daily prices for 10 tickers (8
robotics/automation/deep-tech stocks + SPY/QQQ as broad-market baselines),
then computes three kinds of features per asset per day:

- **Daily return** — how much the price moved that day, as a percentage.
  This is the thing we eventually want to predict.
- **30-day volatility** — how much the daily returns have been bouncing
  around lately, annualized (multiplied by √252 so it's expressed as a
  "per year" number, the market-standard convention). Higher volatility =
  riskier asset.
- **Momentum (30-day and 90-day)** — the cumulative return over the last
  month/quarter. Momentum is one of the most well-documented patterns in
  markets: assets that have recently gone up tend to (weakly) keep going up
  over medium horizons.

**Why these specific features and not others:** They're simple,
interpretable, and don't require any exotic data source. The point of this
project isn't "throw 200 opaque features at a black-box model" — it's to
build something a beginner can fully explain, end to end. Fewer, cleaner
features also reduce the risk of the model finding spurious patterns
(overfitting) in a dataset that isn't huge to begin with.

**A real bug I hit and fixed:** `ABB` (originally planned as a robotics
holding) came back "delisted" from Yahoo's API mid-build, so I swapped in
Parker Hannifin (`PH`) — same industrial-automation theme, live data. This
is a good interview anecdote: real data pipelines break on real-world data
quality issues, and the fix was to verify and substitute, not to force it.

---

## 4. Stage 2 — `return_predictor.py`: the ML model

**What it does:** Trains a separate **Ridge regression** model per ticker
to predict *tomorrow's* return from today's volatility/momentum features.

**What Ridge regression is, in plain English:** It's linear regression
(fit a straight-line-style relationship between features and a target)
with one addition: a penalty that discourages the model from assigning
huge weight to any one feature. Think of it as regression with a built-in
"be humble" dial. That dial is controlled by a number called **alpha** —
higher alpha means more humility (coefficients shrink closer to zero).

**Why Ridge and not plain linear regression:** Our features are weak, noisy
predictors of next-day returns (this is expected — see the Efficient
Market Hypothesis note below). Plain linear regression has no defense
against overfitting to that noise; Ridge's penalty is exactly the right
tool when you suspect there isn't much real signal to find.

**Why Ridge and not something fancier (random forest, neural network):**
With only 3 features and a genuinely weak signal, a more complex model
wouldn't find more real signal — it would just be much better at fitting
noise (overfitting), and far harder to explain and defend. Simplicity here
is a deliberate, defensible choice, not a limitation I ran out of time to
fix.

**The single most important design decision in this file: the time-based
train/test split.** Data is split by date — the earliest 80% of days train
the model, the most recent 20% test it — instead of randomly shuffling
rows into train/test groups.

- **Why this matters (lookahead bias):** if you shuffled randomly, the
  model could train on data from *after* some of the days it's being
  tested on. In the real world, you never get to use tomorrow's newspaper
  to trade today — a random split would let the model implicitly cheat,
  and the reported accuracy would be fake/inflated.
- Hyperparameter selection (choosing `alpha`) uses `TimeSeriesSplit`
  cross-validation for the same reason — even the process of *tuning* the
  model never lets it see the future relative to what it's being validated
  against.

**The honest result:** average test R² across all 10 tickers was **-0.009**
— essentially zero, meaning the model explains almost none of the
variance in next-day returns. Directional accuracy was **50.8%** — a coin
flip. It beat a "just guess the historical average" baseline on only 3 of
10 tickers.

**Why this is a feature of the project, not a failure:** This result is
*expected*. The Efficient Market Hypothesis argues that publicly available
information — like recent momentum and volatility — should already be
priced into a stock, leaving little exploitable signal for anyone,
especially a 3-feature linear model. Reporting this honestly (instead of
tuning until the number looked better, or silently hiding it) is the whole
point of the project's design.

---

## 5. Stage 3 — `optimizer.py`: turning predictions into a portfolio

**What it does:** Given the ML model's predicted returns and the assets'
*historical* covariance (how they've actually moved together), it finds
the portfolio weights that maximize the **Sharpe ratio**:

```
Sharpe ratio = (portfolio return − risk-free rate) / portfolio volatility
```

**Why Sharpe ratio and not just "highest predicted return":** A portfolio
that dumps everything into the single highest-predicted-return asset
technically maximizes return, but ignores risk entirely and gets zero
diversification benefit. Sharpe ratio rewards return *per unit of risk
taken*, which is a far more sensible thing to optimize.

**Why the covariance matrix, not just each asset's own volatility:**
Portfolio risk isn't a simple average of each asset's individual risk —
if two assets tend to move in opposite directions, holding both actually
*reduces* combined risk below what you'd naively expect. The covariance
matrix captures exactly that, mathematically, via `w' · Cov · w` (a
quadratic form). This is the entire mathematical reason diversification
works, and being able to say that sentence out loud is a good interview
signal.

**Why `scipy.optimize.minimize(method='SLSQP')`:** Maximizing Sharpe ratio
subject to "no shorting" (weights between 0 and 1) and "fully invested"
(weights sum to 1) is a constrained nonlinear optimization problem with no
simple closed-form solution once those bounds are added. SLSQP
(Sequential Least Squares Programming) is a standard, well-tested algorithm
for exactly this shape of problem — smooth objective, smooth constraints.

**The efficient frontier:** By sweeping many target returns and, for each,
finding the minimum-risk portfolio that achieves it, you trace out a curve
— the set of portfolios where you can't get more return without accepting
more risk. Individual assets plotted on the same risk/return axes should
sit *below and to the right* of this curve, visually proving
diversification is working.

**A real bug I found and fixed here:** my first version of the frontier
calculation included a mathematically real but *dominated* "lower branch"
— portfolios that take on more risk for a *lower* return than what the
minimum-variance portfolio alone already offers. That's not "efficient" by
definition; a rational investor would never want it. I fixed it by
filtering the frontier to only the branch at or above the global-minimum-
variance point. This mattered beyond just the plot — the dashboard's
risk-tolerance slider picks the nearest frontier point by volatility, and
without the fix it could silently hand back a bad, dominated portfolio.
This is a great interview story about testing your own assumptions: the
bug wasn't caught by unit tests, it was caught by actually using the
feature end-to-end.

**A humbling side effect worth being ready to discuss:** the resulting
max-Sharpe portfolio put 83% of its weight in one stock (HON) with a
suspiciously high Sharpe ratio of 2.67. This happens because portfolio
optimizers are extremely sensitive to their return estimates — feed one
noisy, low-confidence predictions (recall: R² ≈ 0) and it will confidently,
aggressively concentrate around whatever noise happened to look best. This
effect has a name in finance: **estimation-error maximization**. It's a
big part of why Stage 5 (the backtest) exists — a good-looking optimizer
output can be a red flag for overfitting to noisy inputs, not evidence of
a good strategy.

---

## 6. Stage 4 — `monte_carlo.py`: quantifying the downside

**What it does:** Simulates 10,000 possible 1-year futures for the
portfolio by **bootstrapping** — randomly resampling, with replacement,
from the portfolio's actual historical daily returns — and compounding
each sampled sequence into a full price path.

**Why simulate instead of trusting the single expected-return number from
the optimizer:** An "expected return" is an average outcome. It hides the
spread — a portfolio "expected" to return 15%/year can still very
plausibly lose 20% in a bad year, and the expected value alone doesn't
communicate that. Simulating thousands of plausible paths reveals the
whole distribution, not just its center.

**Why bootstrap real historical returns instead of assuming a normal
distribution:** Real markets have "fat tails" — extreme days happen more
often than a bell curve predicts. Resampling actual history preserves
whatever real skew/fat-tailedness existed, which is more honest than
assuming smooth randomness.

**Value at Risk (VaR) and Expected Shortfall, in plain English:**
- **VaR (95%)** answers: "at a 95% confidence level, what's the worst loss
  I should expect over the next year?" It's a single threshold number.
- **Expected Shortfall** answers a follow-up question VaR can't: "*if* I
  land in that worst 5% of outcomes, how bad is it on average?" Two
  portfolios can share the same VaR but have very different tail severity
  beyond it — Expected Shortfall is why regulators increasingly prefer it
  over VaR alone.

**A limitation I'd volunteer, not wait to be asked about:** this bootstrap
samples each day *independently*, which ignores "volatility clustering" —
the real tendency for turbulent days to cluster together (e.g. around a
crash) rather than being scattered randomly. A "block bootstrap" (resampling
contiguous multi-day chunks) would capture that more realistically. I chose
the simpler day-by-day version because it's much easier to explain and
reason about, at the honest cost of understating true crash-clustering
risk.

---

## 7. Stage 5 — `backtest.py`: the moment of truth

This is the file that actually answers the project's core question, and
the one I'd spend the most interview time on.

**What "walk-forward" means and why it's the only honest way to backtest:**
Every earlier stage used the *full* dataset at once — including dates that
are chronologically "future" relative to other dates also used in that
same fit. A walk-forward backtest instead repeatedly: (1) looks only at a
trailing window of history, (2) makes a decision (portfolio weights) using
*only* that window, (3) holds those weights for the next short period and
records what *actually* happened — data the strategy couldn't see when it
decided — then (4) rolls the window forward and repeats. It's the closest
a backtest can get to simulating what would have really happened if you'd
traded this, live, in the past.

**The three strategies compared, and why exactly these three:**
1. **Equal-weight** — 1/N in every asset. No optimization, no prediction.
   The simplest possible baseline; if nothing beats this, that's an
   important, humbling finding on its own.
2. **Naive-mean-optimized** — same SLSQP optimizer, same covariance
   estimate, but fed a *naive* expected return (just the trailing average)
   instead of an ML prediction.
3. **ML-predicted-optimized** — the full pipeline.

Using the *same* optimizer, *same* covariance matrix, and *same*
rebalancing schedule for strategies 2 and 3 isolates one clean variable:
does the ML model's return forecast specifically add value over "just use
the recent average," everything else held equal? That's a fair,
controlled comparison — not just "vibes."

**The honest result (912 out-of-sample trading days, ~3.6 years, monthly
rebalancing):**

| Strategy | Sharpe Ratio | Cumulative Return | Max Drawdown |
|---|---|---|---|
| Equal-weight | 1.22 | 188.6% | -28.4% |
| Naive historical mean | **1.58** | 660.7% | -28.7% |
| ML-predicted | 1.56 | 589.5% | -26.9% |

**How I'd summarize this in an interview:** "The ML strategy clearly beat
doing nothing sophisticated — it comfortably outperformed equal-weight.
But it did *not* beat the much simpler naive-historical-mean approach. That
tracks with what we already knew from Stage 2: the model's R² was
essentially zero. A weak, noisy signal doesn't reliably translate into
better real portfolio performance once it's run through an optimizer that
amplifies whatever it's given. I'd call this an honest negative-ish result,
not a failed project — the pipeline did exactly what it was supposed to
do: it told the truth."

**Why max drawdown matters alongside Sharpe:** Sharpe is an *average*
risk-adjusted measure — a strategy can have a great Sharpe ratio while
still including one terrifying 40% peak-to-trough decline it happened to
recover from. Max drawdown answers "what's the worst peak-to-valley loss
an investor would have actually had to sit through," which is often more
viscerally decision-relevant than an average.

---

## 8. Stage 6 — `app.py`: the dashboard

**What it does:** A Streamlit app with one interactive control — a
1-to-10 "risk tolerance" slider — that maps onto a specific point along
the efficient frontier (1 → lowest-risk frontier point, 10 → highest-risk,
interpolated between). That selected portfolio drives a live efficient
frontier chart, a live-resimulated Monte Carlo fan chart, and a (fixed,
slider-independent) backtest comparison.

**Why the heavy stages are cached (`st.cache_resource`):** Streamlit
reruns the entire script top-to-bottom on every widget interaction.
Without caching, moving the slider would silently redownload data, retrain
10 Ridge models, and rerun the ~6-second walk-forward backtest on every
tick — the caching is what keeps the app feeling instant.

**Why the risk-tolerance slider needed the frontier bug fix from Stage 3:**
its whole job is picking the nearest point on the frontier by volatility.
Before the fix, "nearest by volatility" could land on the dominated lower
branch and hand back a bad portfolio — this dashboard is literally what
surfaced that bug during development.

---

## 9. Glossary — concepts to have crisp one-liners ready for

- **Sharpe ratio** — risk-adjusted return: (return − risk-free rate) ÷
  volatility. Higher is better; rewards return *per unit of risk*.
- **Volatility** — how much an asset's returns bounce around; the standard
  deviation of returns, usually annualized.
- **Covariance / diversification** — how two assets move together;
  negative co-movement between held assets reduces combined portfolio risk
  below a simple average — the mathematical basis of "don't put all your
  eggs in one basket."
- **Momentum** — the tendency for recent winners to keep performing well
  over medium horizons; one of the most persistent, documented patterns in
  equity markets.
- **Efficient frontier** — the set of portfolios offering the best possible
  return for each level of risk; anything below/right of it is dominated
  (strictly worse).
- **Regularization (Ridge/L2)** — a penalty added to a model's loss
  function that shrinks its coefficients toward zero, trading a little
  training accuracy for a lot less overfitting.
- **Lookahead bias** — accidentally letting a model see information from
  the future relative to what it's being tested on; the single most common
  way backtests lie to you.
- **R² (R-squared)** — the fraction of variance in the target a model
  explains; 0 means "no better than always predicting the average," and
  negative means "worse than that."
- **Walk-forward backtest** — repeatedly train/decide on a trailing window,
  test on the next unseen period, then roll forward — the realistic
  alternative to fitting once on all historical data.
- **Value at Risk (VaR)** — the loss threshold you shouldn't expect to
  exceed at a given confidence level (e.g. 95%) over a given horizon.
- **Expected Shortfall** — the average loss *given* that you've landed in
  the worst tail (e.g. worst 5%) of outcomes; a stricter follow-up to VaR.
- **Max drawdown** — the largest peak-to-trough decline in a strategy's
  value over the tested period.
- **Bootstrap** — building a synthetic sample distribution by resampling
  actual observed data with replacement, instead of assuming a theoretical
  distribution (like "normal").
- **Efficient Market Hypothesis (EMH)** — the idea that public information
  is already reflected in prices, which is exactly why simple, public
  features struggle to predict future returns — a useful frame for why
  this project's near-zero R² is expected, not broken.

---

## 10. Likely interview questions, and how I'd answer them

**"Did the ML model actually help?"**
Partially. It beat equal-weight but not a naive historical-mean baseline,
consistent with the model's near-zero R². The honest takeaway is that
weak signal, once run through an optimizer, doesn't reliably translate
into better real performance — and that you only find that out by testing
against a simple baseline out-of-sample, not by trusting a good-looking
in-sample Sharpe ratio.

**"Why Ridge regression and not XGBoost/a neural network?"**
The signal-to-noise ratio in 3 simple technical features predicting daily
stock returns is very low. A more complex model wouldn't find more real
signal in that regime — it would just overfit the noise more
convincingly, and be much harder to explain, debug, and defend. Simplicity
was a deliberate choice matched to the problem, not a shortcut.

**"How did you avoid lookahead bias?"**
Two ways: (1) the ML train/test split is strictly by date, never randomly
shuffled, including during hyperparameter tuning (`TimeSeriesSplit`); (2)
the backtest is walk-forward — at every rebalance step, both the naive-mean
and ML strategies are retrained/re-estimated using only the trailing
window of data available as of that point in time, then scored on
genuinely subsequent, unseen returns.

**"What would you change with more time?"**
Add a block bootstrap to Monte Carlo (to capture volatility clustering),
model transaction costs in the backtest (currently assumes free rebalancing),
expand the feature set carefully (while being alert that more features
don't guarantee more real signal), and test sensitivity to the rolling
window length and rebalancing frequency, which were fixed choices here.

**"What was the hardest bug?"**
The efficient frontier bug (Section 5) — the math was locally correct at
every step (minimize volatility for a target return) but produced a
globally wrong artifact (a dominated lower branch) that only became
obvious once it fed a real downstream feature (the risk-tolerance slider)
and produced a nonsensical -28% "recommended" portfolio. It's a good
example of why testing a pipeline end-to-end, not just its individual
functions, matters.

**"Why not just always use the max-Sharpe portfolio instead of a
risk-tolerance slider?"**
Different investors have different real capacity/willingness to take on
risk. The max-Sharpe portfolio is optimal only under one specific
assumption (maximize return per unit of risk with no other preference) —
a genuinely risk-averse investor might rationally prefer a lower-risk,
lower-Sharpe point on the frontier. The slider makes that trade-off
explicit and adjustable rather than hard-coding one answer for everyone.
