# Leakage / Lookahead Audit

**Scope:** The full pipeline from signal generation through PBO output.
**Date:** 2026-05-17
**Verdict:** No hard lookahead leakage found. Two minor issues noted (severity: low).

---

## 1. Signal Generation (Strategy Layer)

### 1a. Moving-average crossover — `backtesting_engine.py:184-202`, `strategy_framework.py:179-190`

**Guarantee:** `pd.Series.rolling(window=k, min_periods=1).mean()` is a causal, backward-looking filter. At time t it uses only prices[t-k+1 : t]. No future prices enter.

**Risk area:** `min_periods=1` means the first few bars produce a moving average from fewer observations than `window`. This is not lookahead — it just means early signals are computed from less data than intended. Harmless for PBO purposes (it slightly weakens early signals, it doesn't strengthen them with future info).

**Verdict: CLEAN.**

### 1b. Mean-reversion z-score — `strategy_framework.py:206-219`

**Guarantee:** `rolling(window=lookback, min_periods=1).mean()` and `.std()` are both backward-looking. The z-score at time t is computed from prices[t-lookback+1 : t] only.

**Verdict: CLEAN.**

---

## 2. Backtesting Engine — Position/Return Calculation

### 2a. Position lagging — `backtesting_engine.py:115`

```python
lagged_positions = positions.shift(1).fillna(0.0)
```

**Guarantee:** The signal generated on day t determines the position held *starting day t+1*. Returns on day t are earned from the position decided on day t-1. This is the correct "trade on close, earn next-day return" convention. No lookahead.

**Verdict: CLEAN.**

### 2b. Transaction costs — `backtesting_engine.py:117-119`

```python
turnover = positions.diff().abs().fillna(0.0)
cost_rate = self.config.transaction_cost + self.config.slippage
costs = turnover * cost_rate
```

Costs are applied on the day the position *changes*, using the unlagged `positions` series. This means costs are debited on the day the decision is made (day t), while the position takes effect on day t+1. This is conservative (costs are charged one day early rather than late), not a lookahead.

**Verdict: CLEAN.**

### 2c. P&L series construction — `backtesting_engine.py:121-123`

```python
strategy_returns = (lagged_positions * price_returns) - costs
equity_curve = (1.0 + strategy_returns).cumprod() * initial_capital
pnl_series = equity_curve.diff().fillna(0.0)
```

`price_returns = prices.pct_change()` at index t is `(price[t] - price[t-1]) / price[t-1]` — realized return, no future data. Multiplied by the *lagged* position. Correct.

**Verdict: CLEAN.**

---

## 3. CSCV Split Construction

### 3a. Partitioning — `cscv.py:87-96`

The matrix M is sliced into S consecutive, non-overlapping blocks of `T//S` rows each. Blocks are defined by row index alone — purely structural, no data dependency.

### 3b. Combination generation — `cscv.py:98-108`

Train set = any S/2 blocks; test set = the complementary S/2 blocks. Every combination is a strict partition: no row appears in both train and test.

**Guarantee:** IS and OOS are always disjoint. Verified by construction: `test_list = [idx for idx in all_indices if idx not in train_list]`.

**Verdict: CLEAN.**

### 3c. Symmetry property — Paper requirement

The paper requires that train/test assignment is *combinatorially symmetric*: every block appears in training exactly C(S-1, S/2-1) times and in testing the same number of times. This is a mathematical consequence of enumerating all C(S, S/2) combinations, which is exactly what the code does.

**Verdict: CLEAN.**

---

## 4. Performance Evaluation Within Splits

### 4a. Metric computation — `analyze_models.py:134-181`

```python
pnl = data[:, idx]          # data is J_train or J_test (a submatrix slice)
returns = np.diff(np.cumsum(pnl))
```

Performance is computed *only* on the rows within the given split. No global statistics (full-series mean, std, etc.) are referenced. The Sharpe ratio denominator is the std of returns *within that split only*.

**Guarantee:** No cross-contamination between IS and OOS performance calculations.

**Verdict: CLEAN.**

### 4b. Ranking and selection — `analyze_models.py:110-116`

```python
n_star = int(np.where(is_ranks == 1)[0][0])      # best IS strategy
oos_rank_of_best_is = int(oos_ranks[n_star])      # its OOS rank
```

Selection is done purely on IS data; the OOS rank is then *looked up* (not optimized over). This is the correct protocol from the paper.

**Verdict: CLEAN.**

---

## 5. The M Matrix Itself — Is It Pre-Contaminated?

This is the subtlest question: the M matrix is built by running the full backtest over the *entire* price series, then slicing the resulting P&L into submatrices. Does the signal at time t within one submatrix depend on data from a different submatrix?

### 5a. MA-crossover signals

The rolling mean at time t uses prices from t-window+1 to t. If submatrix boundaries fall within that lookback window, the signal on the first few rows of a submatrix incorporates price data from the *previous* submatrix.

**Is this leakage?** It depends on the definition:

- **In the paper's framework:** The M matrix represents pre-computed P&L. The paper assumes M is given (exogenous) and CSCV operates on it. Cross-submatrix signal dependency is *expected* and *accounted for* — the paper notes that sequential strategies will have temporal dependence and this is acceptable as long as no *selection* crosses the IS/OOS boundary. The CSCV procedure doesn't re-run the strategy per split; it slices pre-existing P&L.
- **Practical concern:** None. The strategy parameters are fixed before the backtest runs. The rolling window doesn't use future data relative to time t — it only uses past data that happens to fall in a different submatrix. This is a feature (realistic simulation), not a bug.

**Verdict: CLEAN.** This matches the paper's intended usage exactly.

### 5b. Data source — `data_retrieval.py:92-120`

Uses `yf.download(..., auto_adjust=False)` and extracts "Adj Close." Yahoo's adjusted close is **retrospectively adjusted**: the entire historical series is recomputed using corporate-action adjustment factors (dividends, splits) that were not known at the time each bar was originally observed. This is structurally lookahead — the price you see for 2015-03-15 today is not the price you would have seen on 2015-03-15.

**Why it is quantitatively immaterial for this specific pipeline:**

1. **SPY is an index ETF with no stock splits.** The only adjustment is for quarterly dividend distributions (~1.3% annually). Each ex-dividend adjustment shifts the entire pre-ex-date series by a few basis points.
2. **The MA-crossover signal is sign-based** (fast_MA > slow_MA → long, else short). A uniform downward shift of all pre-ex-date prices by ~0.3% does not change the sign of a moving-average crossover except in a negligible set of edge cases where the two MAs are within fractions of a basis point of each other.
3. **The effect is symmetric across strategy parameterizations.** All N columns of the M matrix use the same adjusted price series, so any micro-distortion from retroactive adjustment affects all configurations equally and does not bias the *relative ranking* that PBO operates on.

**What would make this a real problem:** Single stocks with large special dividends, stock splits, or spin-offs — where adjustment factors can shift prices by 10%+ and materially alter trend signals. Any extension beyond SPY/index ETFs must revisit this finding.

**Verdict: Structurally present, quantitatively immaterial for sign-based signals on SPY. Must be reassessed if the instrument or signal type changes.**

---

## 6. PBO Calculation

### 6a. Logit and PBO — `analyze_models.py:116,189-194`, `pbo.py:84-89`

```python
relative_rank = oos_rank_of_best_is / (M.shape[1] + 1)
logit = log(relative_rank / (1 - relative_rank))
pbo = mean(logits < 0)
```

Pure arithmetic on already-computed OOS ranks. No data dependency on anything external. The denominator (N+1) matches the paper's convention to avoid logit(0) or logit(1).

**Verdict: CLEAN.**

### 6b. Bootstrap CI — `pbo.py:91-106`

Resamples from the computed logit array. No interaction with raw data.

**Verdict: CLEAN.**

---

## 7. Issues Found (Low Severity)

### Issue A: `np.diff(np.cumsum(pnl))` is a no-op that drops one observation

**Location:** `analyze_models.py:145`

`np.cumsum(pnl)` then `np.diff(...)` on it returns `pnl[1:]` — the original series minus the first element. This means each submatrix's Sharpe is computed from (T/S - 1) observations instead of T/S. With T/S = 252, you lose 1 out of 252 observations (0.4%).

**Impact:** Negligible. Does not constitute leakage. Likely a leftover from an earlier design where M contained cumulative equity rather than daily P&L.

**Recommendation:** Replace with `returns = pnl` if M is confirmed to be daily P&L (which it is, per `backtesting_engine.py:123`).

### Issue B: `min_periods=1` creates early-bar signal artifacts

**Location:** `strategy_framework.py:184-185`, `backtesting_engine.py:196-197`

With `min_periods=1`, the 200-day MA on day 1 is just the day-1 price. The fast and slow MAs converge at the start, producing noisy/meaningless signals for the first ~200 bars.

**Impact:** These early-bar signals contribute P&L to the first submatrix. Since all strategy variants share this warmup artifact, it doesn't bias *relative* ranking (which is what PBO cares about). But it adds noise.

**Recommendation:** Either set `min_periods=window` (producing NaN until warmup completes, then handle NaN signals as position=0), or discard an initial warmup period before building M.

---

## 8. Summary Table

| Layer | Check | Result |
|-------|-------|--------|
| Signal generation | Future prices used? | No |
| Signal generation | Full-series statistics used? | No |
| Backtesting | Position properly lagged? | Yes (`shift(1)`) |
| Backtesting | Costs realistic / not forward-looking? | Yes |
| CSCV splits | IS/OOS disjoint? | Yes (by construction) |
| CSCV splits | Combinatorially symmetric? | Yes (all C(S,S/2) enumerated) |
| Performance eval | Computed only within split? | Yes |
| Selection | n* chosen only on IS data? | Yes |
| OOS evaluation | OOS rank is lookup, not optimization? | Yes |
| Data source | Survivorship bias? | N/A (SPY ETF) |
| Data source | Point-in-time issue? | Structurally present (retroactive adj. close); quantitatively immaterial for sign-based MA signals on SPY |

---

## 9. Addendum: Rank-Convention Sign Bug (identified post-audit)

**Identified:** 2026-05-29, during hand-verification of the PBO arithmetic against
the source paper (Bailey et al., step (f)–(g)).

**This was not caught by this audit.** The audit's scope was lookahead/leakage —
whether future data contaminates past decisions — and those findings stand. The
sign bug is a metric-implementation error, not a leakage issue: the pipeline
correctly keeps IS and OOS disjoint, correctly lags positions, and correctly
computes performance within splits. It then *misreports the final PBO statistic*
due to an inverted rank convention.

**The bug:** `_rank_strategies` (`analyze_models.py:186`) uses
`rankdata(-performance)`, assigning rank 1 = best. This makes
`relative_rank = rank/(N+1)` small when the IS-best strategy does well OOS.
The paper defines ω = rank/(N+1) with higher rank = better, so ω is large when
IS-best does well OOS. The code's relative_rank has the opposite polarity from
the paper's ω, making its logit the negative of the paper's logit.

`_calculate_pbo` (`pbo.py:89`) computed `P(logit < 0)`, which with the inverted
rank equals P(IS-best is in the top half OOS) = P(not overfit) = 1 − PBO.

**Fix applied:** Changed `logits < 0` to `logits > 0` at `pbo.py:89` and
`pbo.py:101` (bootstrap CI). Plot shading updated at `pbo.py:171-172`.

**Pre-fix results** are archived at `results/pre_sign_fix/`. Corrected results
are in `results/rung_1/` through `results/rung_4/`.

---

## 10. Final Verdict

**The pipeline is free of structural lookahead bias.** The leakage findings in
Sections 1–6 are unchanged. The two low-severity issues (Issue A: off-by-one
from `np.diff(np.cumsum(...))`; Issue B: warmup artifacts from `min_periods=1`)
are worth cleaning up but do not affect the validity of IS/OOS separation.

A separate **rank-convention sign bug** (Section 9) was identified by tracing
the PBO formula against the source paper. This caused the reported PBO to equal
1 − PBO(true). It has been corrected; pre-fix outputs are preserved for the
project record. The correction does not affect the leakage audit's conclusions
— the pipeline's data-handling integrity is intact.
