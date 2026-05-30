"""Rules ladder: PBO vs. parameter-search-space dimensionality.

Runs the same MA-crossover strategy form at four rungs of increasing
parameter-space size, holding everything else constant (data, engine,
costs, CSCV config).  Produces four PBO estimates and a summary figure.

Usage:
    python rules_ladder.py              # run all four rungs
    python rules_ladder.py --rungs 1 2  # run specific rungs only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_models import (
    AnalysisConfig,
    ModelAnalysisResult,
    ModelAnalyzer,
    PerformanceMetric,
)
from backtesting_engine import BacktestConfig, BacktestEngine
from cscv import CSCVConfig, CSCVPartitioner
from pbo import PBOCalculator, PBOConfig

logger = logging.getLogger(__name__)

S = 16
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Signal function
# ---------------------------------------------------------------------------

def ladder_ma_signal(
    price_data: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    confirmation_days: int = 1,
    stop_loss_pct: Optional[float] = None,
    holding_period: int = 1,
) -> pd.Series:
    """MA crossover with optional confirmation, stop-loss, and minimum hold.

    Parameters
    ----------
    price_data : DataFrame passed by BacktestEngine (single-column expected).
    fast, slow : MA window lengths.
    confirmation_days : consecutive bars a crossover must persist before acting.
        1 = immediate (reproduces base MA crossover).
    stop_loss_pct : fraction (e.g. 0.02 for 2%).  None = no stop.
    holding_period : minimum bars to hold before allowing a position change.
        1 = can change every bar (reproduces base behavior).
    """
    price_series = price_data.iloc[:, 0]
    prices_arr = price_series.to_numpy(dtype=float)
    n = len(prices_arr)

    fast_ma = price_series.rolling(window=fast, min_periods=1).mean().to_numpy()
    slow_ma = price_series.rolling(window=slow, min_periods=1).mean().to_numpy()

    # Raw directional signal at each bar: +1 / -1 / 0
    raw = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))

    signals = np.zeros(n, dtype=float)
    pos = 0.0
    hold_count = 0
    entry_price = 0.0
    prev_raw = 0.0
    consec = 0

    for i in range(n):
        r = raw[i]
        price = prices_arr[i]

        # --- Confirmation tracking ---
        if r != 0 and r == prev_raw:
            consec += 1
        elif r != 0:
            consec = 1
        else:
            consec = 0
        prev_raw = r

        confirmed = r if consec >= confirmation_days else 0.0

        # --- Stop-loss check (before position update) ---
        if pos != 0 and stop_loss_pct is not None:
            if pos > 0 and price <= entry_price * (1.0 - stop_loss_pct):
                pos = 0.0
                hold_count = 0
            elif pos < 0 and price >= entry_price * (1.0 + stop_loss_pct):
                pos = 0.0
                hold_count = 0

        # --- Position update ---
        if pos == 0.0:
            # Flat: enter on confirmed signal
            if confirmed != 0.0:
                pos = confirmed
                entry_price = price
                hold_count = 1
        else:
            hold_count += 1
            # In position: only allow change after holding period
            if hold_count > holding_period and confirmed != 0.0 and confirmed != pos:
                pos = confirmed
                entry_price = price
                hold_count = 1

        signals[i] = pos

    return pd.Series(signals, index=price_series.index, dtype=float)


# ---------------------------------------------------------------------------
# Rung grid definitions
# ---------------------------------------------------------------------------

BASE_PAIRS: List[Tuple[int, int]] = [
    (10, 30), (10, 50), (20, 50), (20, 100), (30, 100),
    (50, 100), (50, 200), (10, 100), (30, 200), (20, 200),
]


def _build_grid(
    confirmation_values: List[int],
    stop_values: List[Optional[float]],
    holding_values: List[int],
) -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    for fast, slow in BASE_PAIRS:
        for conf in confirmation_values:
            for stop in stop_values:
                for hold in holding_values:
                    grid.append({
                        "fast": fast,
                        "slow": slow,
                        "confirmation_days": conf,
                        "stop_loss_pct": stop,
                        "holding_period": hold,
                    })
    return grid


RUNG_GRIDS: Dict[int, List[Dict[str, Any]]] = {
    1: _build_grid([1], [None], [1]),                                             # 10
    2: _build_grid([1, 2, 3, 5, 7], [None], [1]),                                # 50
    3: _build_grid([1, 3, 5], [0.005, 0.01, 0.02, 0.05, None], [1]),             # 150
    4: _build_grid([1, 3, 5], [0.01, 0.02, 0.05, None], [1, 5, 10, 20, 40]),     # 600
}

RUNG_PARAM_NAMES: Dict[int, List[str]] = {
    1: ["fast", "slow"],
    2: ["fast", "slow", "confirmation_days"],
    3: ["fast", "slow", "confirmation_days", "stop_loss_pct"],
    4: ["fast", "slow", "confirmation_days", "stop_loss_pct", "holding_period"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_spy_prices() -> pd.DataFrame:
    """Load SPY from the existing prices.csv."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "prices.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found.  Run data_retrieval.py first to fetch price data."
        )
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "SPY" not in prices.columns:
        raise ValueError("SPY column not found in prices.csv")
    spy = prices[["SPY"]].dropna()
    return spy.sort_index()


def _trim_to_divisible(df: pd.DataFrame, divisor: int) -> pd.DataFrame:
    remainder = len(df) % divisor
    if remainder == 0:
        return df
    return df.iloc[:-remainder]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_m_matrix(
    prices_df: pd.DataFrame,
    grid: List[Dict[str, Any]],
) -> np.ndarray:
    """Run backtest for every config in *grid*, return (T, N) P&L matrix."""
    engine = BacktestEngine(BacktestConfig())
    pnl_columns: List[np.ndarray] = []

    for params in tqdm(grid, desc="  Backtesting configs"):
        result = engine.run(
            price_data=prices_df,
            strategy_func=ladder_ma_signal,
            strategy_params=params,
            price_col="SPY",
        )
        pnl_columns.append(result.pnl_series.to_numpy())

    M = np.column_stack(pnl_columns)
    if np.isnan(M).any():
        raise ValueError("M matrix contains NaNs")
    return M


def run_pbo_analysis(M: np.ndarray) -> Tuple[PBOCalculator, ModelAnalysisResult, Any]:
    """Run CSCV partitioning + model analysis + PBO on matrix M."""
    partitioner = CSCVPartitioner(M, CSCVConfig(S=S))
    analyzer = ModelAnalyzer(
        AnalysisConfig(performance_metric=PerformanceMetric.SHARPE_RATIO)
    )

    combination_results = []
    for split in tqdm(partitioner, total=len(partitioner), desc="  CSCV combinations"):
        combination_results.append(analyzer._analyze_single_combination(split, M))

    logits = np.array([r.logit for r in combination_results], dtype=float)
    logit_distribution = analyzer._build_logit_distribution(logits)

    analysis_result = ModelAnalysisResult(
        combination_results=combination_results,
        logits=logits,
        logit_distribution=logit_distribution,
        n_strategies=M.shape[1],
        n_combinations=len(partitioner),
        metric_used=PerformanceMetric.SHARPE_RATIO,
    )

    calculator = PBOCalculator(PBOConfig())
    pbo_result = calculator.calculate(analysis_result)
    return calculator, analysis_result, pbo_result


def save_rung_results(
    rung: int,
    M: np.ndarray,
    grid: List[Dict[str, Any]],
    prices_df: pd.DataFrame,
    calculator: PBOCalculator,
    analysis_result: ModelAnalysisResult,
    pbo_result: Any,
) -> str:
    """Save all outputs for a single rung.  Returns the rung directory path."""
    rung_dir = os.path.join(RESULTS_DIR, f"rung_{rung}")
    os.makedirs(rung_dir, exist_ok=True)

    # M matrix
    np.save(os.path.join(rung_dir, "M_matrix.npy"), M)

    # Metadata
    metadata = {
        "rung": rung,
        "n_params": len(RUNG_PARAM_NAMES[rung]),
        "param_names": RUNG_PARAM_NAMES[rung],
        "n_configs": len(grid),
        "grid": grid,
        "T": int(M.shape[0]),
        "N": int(M.shape[1]),
        "S": S,
        "date_range": {
            "start": prices_df.index.min().strftime("%Y-%m-%d"),
            "end": prices_df.index.max().strftime("%Y-%m-%d"),
        },
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(rung_dir, "M_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Analysis + PBO pickles
    with open(os.path.join(rung_dir, "analysis_result.pkl"), "wb") as f:
        pickle.dump(analysis_result, f)
    with open(os.path.join(rung_dir, "pbo_result.pkl"), "wb") as f:
        pickle.dump(pbo_result, f)

    # Diagnostic plots
    calculator.plot_all_diagnostics(
        analysis_result, pbo_result,
        save_path=os.path.join(rung_dir, "pbo_diagnostics.png"),
    )

    # Text report
    report = calculator.generate_report(analysis_result, pbo_result)
    with open(os.path.join(rung_dir, "pbo_report.txt"), "w") as f:
        f.write(report)

    return rung_dir


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_pbo_vs_complexity(
    rung_results: Dict[int, Dict[str, Any]],
    save_path: str,
) -> None:
    """PBO (with 95% CI) vs. number of configurations."""
    rungs = sorted(rung_results.keys())
    n_configs = [rung_results[r]["n_configs"] for r in rungs]
    pbos = [rung_results[r]["pbo"] for r in rungs]
    ci_lo = [rung_results[r]["ci_lower"] for r in rungs]
    ci_hi = [rung_results[r]["ci_upper"] for r in rungs]

    yerr_lo = [p - l for p, l in zip(pbos, ci_lo)]
    yerr_hi = [h - p for p, h in zip(pbos, ci_hi)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        n_configs, pbos, yerr=[yerr_lo, yerr_hi],
        fmt="o-", capsize=6, color="#2c3e50", markersize=8, linewidth=1.5,
    )

    for i, r in enumerate(rungs):
        n_params = len(RUNG_PARAM_NAMES[r])
        ax.annotate(
            f"Rung {r}\n{n_params} params, N={n_configs[i]}\nPBO={pbos[i]:.3f}",
            xy=(n_configs[i], pbos[i]),
            xytext=(12, 10), textcoords="offset points",
            fontsize=8, ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Strategy Configurations (N)", fontsize=11)
    ax.set_ylabel("Probability of Backtest Overfitting", fontsize=11)
    ax.set_title(
        "PBO vs. Parameter Search-Space Size\n"
        "Same MA-crossover form, same data, same costs — only grid size varies",
        fontsize=12,
    )
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Rules ladder PBO experiment")
    parser.add_argument(
        "--rungs", type=int, nargs="*", default=[1, 2, 3, 4],
        help="Which rungs to run (default: all four)",
    )
    args = parser.parse_args(argv)

    # --- Load and prepare data (same for every rung) ---
    spy = _load_spy_prices()
    spy = _trim_to_divisible(spy, S)
    T = len(spy)
    logger.info(
        "Data: SPY %s to %s, T=%d, S=%d, submatrix=%d days",
        spy.index.min().strftime("%Y-%m-%d"),
        spy.index.max().strftime("%Y-%m-%d"),
        T, S, T // S,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rung_results: Dict[int, Dict[str, Any]] = {}

    for rung in sorted(args.rungs):
        grid = RUNG_GRIDS[rung]
        n_params = len(RUNG_PARAM_NAMES[rung])
        print(f"\n{'=' * 60}")
        print(f"Rung {rung}: {n_params} params, {len(grid)} configurations")
        print(f"{'=' * 60}")

        t0 = time.time()

        # Build M matrix
        M = build_m_matrix(spy, grid)
        logger.info("M matrix shape: %s", M.shape)

        # Run PBO
        calculator, analysis_result, pbo_result = run_pbo_analysis(M)

        elapsed = time.time() - t0

        # Save
        rung_dir = save_rung_results(
            rung, M, grid, spy, calculator, analysis_result, pbo_result,
        )

        # Report
        print(f"\n  PBO = {pbo_result.pbo:.4f}  "
              f"[{pbo_result.pbo_lower_ci:.4f}, {pbo_result.pbo_upper_ci:.4f}]")
        print(f"  Performance degradation: {pbo_result.performance_degradation:.4f}")
        print(f"  Saved to: {rung_dir}")
        print(f"  Elapsed: {elapsed:.1f}s")

        rung_results[rung] = {
            "n_configs": len(grid),
            "pbo": pbo_result.pbo,
            "ci_lower": pbo_result.pbo_lower_ci,
            "ci_upper": pbo_result.pbo_upper_ci,
            "degradation": pbo_result.performance_degradation,
        }

    # --- Summary ---
    if len(rung_results) > 1:
        summary_path = os.path.join(RESULTS_DIR, "pbo_vs_complexity.png")
        plot_pbo_vs_complexity(rung_results, summary_path)
        print(f"\nSummary figure saved to: {summary_path}")

    print(f"\n{'=' * 60}")
    print("LADDER SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Rung':<6} {'Params':<8} {'N':>6} {'PBO':>8} {'95% CI':>20} {'Degradation':>13}")
    print("-" * 65)
    for rung in sorted(rung_results):
        r = rung_results[rung]
        print(
            f"{rung:<6} {len(RUNG_PARAM_NAMES[rung]):<8} {r['n_configs']:>6} "
            f"{r['pbo']:>8.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] "
            f"{r['degradation']:>13.4f}"
        )

    # Save summary as JSON for downstream use
    summary_json_path = os.path.join(RESULTS_DIR, "ladder_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(rung_results, f, indent=2)


if __name__ == "__main__":
    main()
