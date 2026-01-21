"""Model analysis module for CSCV in-sample vs out-of-sample evaluation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from cscv import CSCVConfig, CSCVPartitioner, CSCVSplit


logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Supported performance metrics."""

    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for model analysis."""

    performance_metric: PerformanceMetric = PerformanceMetric.SHARPE_RATIO
    risk_free_rate: float = 0.0
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class CombinationResult:
    """Results for a single CSCV combination."""

    combination_id: int
    is_performance: np.ndarray
    oos_performance: np.ndarray
    is_ranks: np.ndarray
    oos_ranks: np.ndarray
    n_star: int
    oos_rank_of_best_is: int
    relative_rank: float
    logit: float


@dataclass(frozen=True)
class ModelAnalysisResult:
    """Aggregate results across all CSCV combinations."""

    combination_results: List[CombinationResult]
    logits: np.ndarray
    logit_distribution: Dict[float, float]
    n_strategies: int
    n_combinations: int
    metric_used: PerformanceMetric


class ModelAnalyzer:
    """Analyze strategy performance across CSCV splits."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def analyze_all_splits(self, partitioner: CSCVPartitioner, M: np.ndarray) -> ModelAnalysisResult:
        """Analyze all CSCV splits and return aggregated results."""

        validate_pnl_matrix(M)
        combination_results: List[CombinationResult] = []

        for idx, split in enumerate(partitioner):
            if idx % 100 == 0:
                logger.info("Processing combination %s/%s", idx + 1, len(partitioner))
            combination_results.append(self._analyze_single_combination(split, M))

        logits = np.array([result.logit for result in combination_results], dtype=float)
        logit_distribution = self._build_logit_distribution(logits)

        if logits.size:
            logger.info(
                "Logit summary: mean=%.4f std=%.4f min=%.4f max=%.4f",
                float(np.mean(logits)),
                float(np.std(logits, ddof=0)),
                float(np.min(logits)),
                float(np.max(logits)),
            )

        return ModelAnalysisResult(
            combination_results=combination_results,
            logits=logits,
            logit_distribution=logit_distribution,
            n_strategies=M.shape[1],
            n_combinations=len(partitioner),
            metric_used=self.config.performance_metric,
        )

    def _analyze_single_combination(self, split: CSCVSplit, M: np.ndarray) -> CombinationResult:
        """Analyze a single CSCV combination and return metrics."""

        is_performance = self._calculate_performance(split.J_train)
        oos_performance = self._calculate_performance(split.J_test)

        is_ranks = self._rank_strategies(is_performance)
        oos_ranks = self._rank_strategies(oos_performance)

        n_star = int(np.where(is_ranks == 1)[0][0])
        oos_rank_of_best_is = int(oos_ranks[n_star])

        relative_rank = oos_rank_of_best_is / (M.shape[1] + 1)
        logit = self._calculate_logit(relative_rank)

        if abs(logit) > 10:
            logger.warning("Extreme logit value detected: %.4f", logit)

        return CombinationResult(
            combination_id=split.combination_id,
            is_performance=is_performance,
            oos_performance=oos_performance,
            is_ranks=is_ranks,
            oos_ranks=oos_ranks,
            n_star=n_star,
            oos_rank_of_best_is=oos_rank_of_best_is,
            relative_rank=relative_rank,
            logit=logit,
        )

    def _calculate_performance(self, data: np.ndarray) -> np.ndarray:
        """Calculate the configured performance metric for each strategy."""

        if data.ndim != 2:
            raise ValueError("data must be a 2D array")

        metric = self.config.performance_metric
        performance = np.zeros(data.shape[1], dtype=float)

        for idx in range(data.shape[1]):
            pnl = data[:, idx]
            returns = np.diff(np.cumsum(pnl))
            if returns.size == 0:
                performance[idx] = 0.0
                continue

            if metric == PerformanceMetric.SHARPE_RATIO:
                excess = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
                denom = np.std(excess, ddof=0)
                performance[idx] = 0.0 if denom == 0 else np.mean(excess) / denom * math.sqrt(
                    self.config.trading_days_per_year
                )
            elif metric == PerformanceMetric.TOTAL_RETURN:
                initial_capital = max(abs(np.cumsum(pnl)[0]), 1.0)
                performance[idx] = np.sum(pnl) / initial_capital
            elif metric == PerformanceMetric.SORTINO_RATIO:
                excess = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
                downside = excess[excess < 0]
                downside_dev = np.std(downside, ddof=0)
                performance[idx] = (
                    0.0
                    if downside_dev == 0
                    else np.mean(excess) / downside_dev * math.sqrt(self.config.trading_days_per_year)
                )
            elif metric == PerformanceMetric.CALMAR_RATIO:
                equity = 1.0 + np.cumsum(pnl)
                running_max = np.maximum.accumulate(equity)
                drawdowns = equity / running_max - 1.0
                max_drawdown = abs(drawdowns.min())
                total_return = equity[-1] - 1.0
                annual_return = (1.0 + total_return) ** (
                    self.config.trading_days_per_year / max(len(equity) - 1, 1)
                ) - 1.0
                performance[idx] = 0.0 if max_drawdown == 0 else annual_return / max_drawdown
            else:
                raise ValueError(f"Unsupported performance metric: {metric}")

        return performance

    def _rank_strategies(self, performance: np.ndarray) -> np.ndarray:
        """Rank strategies by performance in descending order (1=best)."""

        ranks = rankdata(-performance, method="ordinal")
        return ranks.astype(int)

    def _calculate_logit(self, relative_rank: float) -> float:
        """Calculate logit for a relative rank."""

        epsilon = 1e-10
        adjusted = min(max(relative_rank, epsilon), 1.0 - epsilon)
        return float(math.log(adjusted / (1.0 - adjusted)))

    def _build_logit_distribution(self, logits: np.ndarray, n_bins: int = 50) -> Dict[float, float]:
        """Build a histogram distribution of logit values."""

        if logits.size == 0:
            return {}

        counts, bin_edges = np.histogram(logits, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = counts.sum()
        distribution = {
            float(center): float(count) / float(total) for center, count in zip(bin_centers, counts)
        }

        logger.info(
            "Logit distribution built with %s bins; total mass=%.4f",
            n_bins,
            float(sum(distribution.values())),
        )
        return distribution


def calculate_pnl_statistics(pnl_series: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for a P&L series."""

    # returns = np.diff(np.cumsum(pnl_series))
    returns = pnl_series
    if returns.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sharpe": 0.0, "total_return": 0.0}

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    sharpe = 0.0 if std == 0 else mean / std * math.sqrt(252)
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "sharpe": sharpe,
        "total_return": float(np.sum(pnl_series)),
    }


def validate_pnl_matrix(M: np.ndarray) -> None:
    """Validate a P&L matrix for use in analysis."""

    if not isinstance(M, np.ndarray) or M.ndim != 2:
        raise ValueError("M must be a 2D numpy array")
    if np.isnan(M).any() or np.isinf(M).any():
        raise ValueError("M contains NaNs or Infs")

    column_sums = np.sum(M, axis=0)
    if np.allclose(column_sums, column_sums[0]):
        raise ValueError("All strategies appear to have identical performance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    T = 1000
    N = 10
    S = 8
    rng = np.random.default_rng(7)
    M = rng.normal(scale=0.5, size=(T, N))

    partitioner = CSCVPartitioner(M, config=CSCVConfig(S=S))

    analyzer = ModelAnalyzer(AnalysisConfig(performance_metric=PerformanceMetric.SHARPE_RATIO))
    result = analyzer.analyze_all_splits(partitioner, M)

    logits = result.logits
    mean_logit = float(np.mean(logits)) if logits.size else 0.0
    std_logit = float(np.std(logits, ddof=0)) if logits.size else 0.0
    pct_negative = float(np.mean(logits < 0) * 100.0) if logits.size else 0.0

    print(f"Mean logit: {mean_logit:.4f}")
    print(f"Std logit: {std_logit:.4f}")
    print(f"% negative logits: {pct_negative:.2f}%")

    if logits.size:
        plt.hist(logits, bins=50, density=True, alpha=0.7)
        plt.title("Logit Distribution")
        plt.xlabel("Logit")
        plt.ylabel("Density")
        plt.show()
