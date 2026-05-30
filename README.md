# Probability of Backtest Overfitting

**Does the size of a strategy's parameter search space predict overfitting risk?**

A Python implementation of the Probability of Backtest Overfitting (PBO) test from Bailey, Borwein, López de Prado & Zhu (2017), via Combinatorially Symmetric Cross-Validation (CSCV). Rather than reproducing the paper on a single strategy, this project uses PBO as a measurement instrument: it builds a **controlled complexity ladder** of MA-crossover configurations on SPY (2010–2026) and asks how overfitting risk responds as the search space grows.

Everything is held fixed across rungs — strategy form, data, transaction costs, and CSCV configuration (S=16, C(16,8)=12,870 combinations). The only thing that changes is the number of tunable parameters, and therefore the number of configurations the optimizer gets to pick from.

---

## Result

| Rung | Parameters varied         | N configs | Beats median | PBO  |
|------|---------------------------|-----------|--------------|------|
| 1    | fast, slow                | 10        | 44%          | 0.56 |
| 2    | + confirmation threshold  | 50        | 57%          | 0.43 |
| 3    | + stop-loss / exit        | 150       | 60%          | 0.37 |
| 4    | + holding period          | 600       | 43%          | 0.57 |

**PBO does not rise monotonically with the size of the search space.** The summary figure is at `results/pbo_vs_complexity.png`.

This is the result I did not expect. The intuition going in was the textbook one — more knobs to turn means more ways to fit noise, so PBO should climb as you walk up the ladder. Instead, adding parameters at Rungs 2 and 3 *reduced* overfitting risk (PBO fell from 0.56 to 0.37), and only at Rung 4 did it jump back up. The interesting question became *why*, and most of the work went into answering it.

---

## Why the curve is non-monotonic

The shape splits into two regimes.

**Rungs 1–3: more structure helps.** The early additions — a confirmation threshold, then a stop-loss/exit — are not free parameters in the "more noise to fit" sense. They constrain the strategy toward economically sensible behavior (don't act on a marginal crossover; cut losers). Selecting the in-sample best within these richer grids produced configurations that *generalized better* out-of-sample, not worse. The beats-median rate rises in lockstep (44% → 57% → 60%), corroborating the PBO drop with a metric that has no dependence on the search-space size.

**Rung 4: the holding-period knob breaks it.** Adding a holding-period parameter pushes the grid to 600 configurations and reverses the trend — PBO 0.57, the worst on the ladder, and the only rung with negative raw OOS Sharpe (−0.068). The holding period is the kind of parameter that lets the optimizer fit the specific *timing* of the in-sample period: it creates configurations that look excellent in-sample for reasons that don't carry forward.

The natural worry is that this jump is an artifact rather than a real effect — so I tried to kill it before believing it.

### The hypothesis I spent the most time on: collapsed effective diversity

When N jumps from 150 to 600, an obvious suspicion is that the extra configurations are **near-duplicates**. If the holding-period knob mostly produces redundant copies of the same handful of underlying strategies, then N=600 is a fiction: the optimizer is really choosing among far fewer distinct things, and that redundancy could mechanically distort PBO. Under this story, the Rung 4 spike isn't telling you anything about overfitting — it's an inflated-N illusion.

This is a clean, testable hypothesis, so I tested it directly with two measures of *effective* (not nominal) diversity, computed from the eigenvalue spectrum of the strategy correlation matrix:

- **Participation ratio** — `(Σλᵢ)² / Σλᵢ²`, the effective number of independent directions in the strategy set. A grid of identical strategies collapses to ~1; a grid of genuinely distinct strategies spreads across many.
- **Components to 95% variance** — how many principal components are needed to capture 95% of the cross-sectional variance.

If the redundancy story were right, Rung 4 should have the *lowest* values on both — many configs, little real variety.

It had the **highest** on both. Rung 4 carried the most effective dimensions and required the most principal components to reach 95% variance. The 600 configurations were *less* redundant than the smaller grids, not more. The holding-period knob was generating genuinely distinct strategies — and those genuinely distinct strategies were genuinely overfit.

That refutation is the substantive point of the project. The Rung 4 PBO spike is **not** a counting artifact of an inflated N. It is a real signal that a particular kind of parameter — one that lets the optimizer fit the timing of the sample — degrades out-of-sample generalization even though the strategies it produces are diverse and individually attractive in-sample. Search-space *size* is the wrong thing to watch; search-space *content* is what matters.

### A second hypothesis, also refuted

I briefly suspected a mechanical "denominator effect": since the relative rank is `ω = rank / (N+1)`, maybe the logit transform `log(ω / (1−ω))` behaves differently at large N and compresses PBO. This dissolves on inspection — `logit(x) < 0` if and only if `x < 0.5`, regardless of N. There is no N-dependence in the PBO threshold. Worth recording because it was part of the actual path, but it didn't survive contact with the algebra.

---

## Methodology notes

Two checks underpin the numbers above; both are documented more fully in the repo.

**Leakage audit** ([`LEAKAGE_AUDIT.md`](LEAKAGE_AUDIT.md)) — verdict: no lookahead leakage. Positions are lagged via `shift(1)`, CSCV IS/OOS partitions are disjoint by construction, and metrics are computed within-split. No survivorship issues for SPY.

**A sign-error catch.** While validating the pipeline I traced the PBO calculation line-by-line against Bailey et al.'s steps (e–g) and found a rank-convention error: the code used `rankdata(-performance)` (rank 1 = best), inverting the polarity relative to the paper's convention (higher rank = better OOS). The net effect was that it computed and reported `1 − PBO`. The fix was a single character (`pbo.py:89`, `<` → `>`); pre-fix outputs are kept at `results/pre_sign_fix/` as history. All numbers in this README are post-correction. (This was a bug in my implementation, not a limitation of PBO — the metric works as designed once the rank convention matches the paper.)

---

## How to reproduce

Python 3.10+.

```bash
pip install -r requirements.txt

python data_retrieval.py     # fetch SPY adjusted close into data/prices.csv (T trimmed to 4032, divisible by S=16)
python rules_ladder.py       # run all four rungs; --rungs 1 2 to run a subset
```

Outputs land in `results/rung_{1,2,3,4}/` (PBO report, diagnostics, the (T,N) P&L matrix and metadata), with the ladder summary at `results/pbo_vs_complexity.png` and `results/ladder_summary.json`.

---

## What I took away

Implementing CSCV/PBO was the easy part — it's matrix bookkeeping following a fixed procedure. The value was in treating PBO as an experiment rather than a number to report: holding the strategy form fixed and varying only complexity, getting a result that contradicted my prior, and then refusing to accept the convenient explanation for it. Watching the "it's just duplicate strategies" hypothesis die against the participation-ratio numbers — and the sign error die against the source paper — was a better lesson in quant discipline than any clean positive result would have been.

---

## Reference

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). *The Probability of Backtest Overfitting*. Journal of Computational Finance, 20(4). PDF included in this repository as `backtest-prob-paper.pdf`.

---

**Questions?** Open an issue or reach out at rishabh.chhabra2024@gmail.com.
