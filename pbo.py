"""Probability of Backtest Overfitting (PBO) diagnostics and reporting."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analyze_models import CombinationResult, ModelAnalysisResult, PerformanceMetric


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PBOConfig:
    """Configuration for PBO calculation and diagnostics."""

    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    plot_style: str = "seaborn-v0_8-darkgrid"
    figure_size: Tuple[int, int] = (12, 8)


@dataclass(frozen=True)
class PBOResult:
    """Results of PBO calculation and diagnostics."""

    pbo: float
    pbo_lower_ci: float
    pbo_upper_ci: float
    performance_degradation: float
    sd2_statistic: float
    n_combinations: int
    n_strategies: int
    metric_used: str
    interpretation: str


class PBOCalculator:
    """Calculate PBO and generate diagnostic plots."""

    def __init__(self, config: PBOConfig) -> None:
        self.config = config

    def calculate(self, analysis_result: ModelAnalysisResult) -> PBOResult:
        """Calculate PBO and supporting diagnostics."""

        logits = analysis_result.logits
        n_combinations = analysis_result.n_combinations

        if n_combinations < 100:
            logger.warning("Number of combinations (%s) may be too small for reliable PBO", n_combinations)

        pbo_value = self._calculate_pbo(logits)
        ci_lower, ci_upper = self._calculate_pbo_confidence_interval(logits)
        perf_degradation = self._calculate_performance_degradation(analysis_result)
        sd2_statistic = self._calculate_stochastic_dominance(analysis_result)

        if logits.size:
            skewness = _calculate_skewness(logits)
            if abs(skewness) > 1.0:
                logger.warning("Logit distribution is heavily skewed (skew=%.3f)", skewness)

        interpretation = interpret_pbo(pbo_value)

        return PBOResult(
            pbo=pbo_value,
            pbo_lower_ci=ci_lower,
            pbo_upper_ci=ci_upper,
            performance_degradation=perf_degradation,
            sd2_statistic=sd2_statistic,
            n_combinations=n_combinations,
            n_strategies=analysis_result.n_strategies,
            metric_used=str(analysis_result.metric_used.value),
            interpretation=interpretation,
        )

    def _calculate_pbo(self, logits: np.ndarray) -> float:
        """Calculate the probability of backtest overfitting."""

        if logits.size == 0:
            return float("nan")
        return float(np.mean(logits < 0))

    def _calculate_pbo_confidence_interval(self, logits: np.ndarray) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the PBO."""

        if logits.size == 0:
            return (float("nan"), float("nan"))

        rng = np.random.default_rng(123)
        pbo_samples = []
        for _ in range(self.config.n_bootstrap_samples):
            sample = rng.choice(logits, size=logits.size, replace=True)
            pbo_samples.append(np.mean(sample < 0))

        alpha = (1.0 - self.config.confidence_level) / 2.0
        lower = float(np.percentile(pbo_samples, 100 * alpha))
        upper = float(np.percentile(pbo_samples, 100 * (1 - alpha)))
        return lower, upper

    def _calculate_performance_degradation(self, analysis_result: ModelAnalysisResult) -> float:
        """Calculate median IS vs OOS performance degradation for best IS strategies."""

        is_perf, oos_perf = extract_best_is_performances(analysis_result)
        if is_perf.size == 0 or oos_perf.size == 0:
            return float("nan")
        return float(np.median(is_perf) - np.median(oos_perf))

    def _calculate_stochastic_dominance(self, analysis_result: ModelAnalysisResult) -> float:
        """Compute second-order stochastic dominance (SD2) statistic."""

        is_perf, oos_perf = extract_best_is_performances(analysis_result)
        if is_perf.size == 0 or oos_perf.size == 0:
            return float("nan")

        values = np.sort(np.unique(np.concatenate([is_perf, oos_perf])))
        if values.size == 0:
            return float("nan")

        cdf_is = np.searchsorted(np.sort(is_perf), values, side="right") / is_perf.size
        cdf_oos = np.searchsorted(np.sort(oos_perf), values, side="right") / oos_perf.size
        sd2 = np.cumsum(cdf_is - cdf_oos)
        return float(sd2[-1])

    def plot_all_diagnostics(
        self,
        analysis_result: ModelAnalysisResult,
        pbo_result: PBOResult,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a 1x3 grid of diagnostic plots."""

        plt.style.use(self.config.plot_style)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        self.plot_logit_distribution(analysis_result.logits, pbo_result.pbo, ax=axes[0])
        self.plot_performance_degradation_scatter(analysis_result, ax=axes[1])
        self.plot_stochastic_dominance(analysis_result, ax=axes[2])

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    def plot_logit_distribution(
        self,
        logits: np.ndarray,
        pbo: float,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Plot histogram of logit values with PBO shading."""

        plt.style.use(self.config.plot_style)
        fig, plot_ax = _get_ax(ax, self.config.figure_size)

        if logits.size == 0:
            plot_ax.text(0.5, 0.5, "No logits available", ha="center", va="center")
        else:
            sns.histplot(logits, bins=50, color="green", edgecolor="black", ax=plot_ax)
            plot_ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            if np.min(logits) < 0:
                plot_ax.axvspan(min(logits), 0, color="red", alpha=0.2)
            mean = np.mean(logits)
            std = np.std(logits, ddof=0)
            if std > 0:
                x_vals = np.linspace(min(logits), max(logits), 200)
                pdf = (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
                pdf_scaled = pdf * (len(logits) * (x_vals[1] - x_vals[0]))
                plot_ax.plot(x_vals, pdf_scaled, color="blue", linewidth=1)

            plot_ax.text(
                0.02,
                0.95,
                f"Prob Overfit = {pbo:.2f}",
                transform=plot_ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        plot_ax.set_title("Histogram of Rank Logits")
        plot_ax.set_xlabel("Logits")
        plot_ax.set_ylabel("Frequency")
        plot_ax.grid(True)

        _finalize_plot(fig, save_path)

    def plot_performance_degradation_scatter(
        self,
        analysis_result: ModelAnalysisResult,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Scatter plot of IS vs OOS Sharpe ratios for best IS strategy."""

        plt.style.use(self.config.plot_style)
        fig, plot_ax = _get_ax(ax, self.config.figure_size)

        is_perf, oos_perf = extract_best_is_performances(analysis_result)
        if is_perf.size == 0:
            plot_ax.text(0.5, 0.5, "No performance data", ha="center", va="center")
        else:
            plot_ax.scatter(is_perf, oos_perf, color="#6b2f2f", alpha=0.8, edgecolor="black")
            if is_perf.size >= 2:
                slope, intercept = np.polyfit(is_perf, oos_perf, 1)
                x_vals = np.linspace(is_perf.min(), is_perf.max(), 100)
                y_vals = intercept + slope * x_vals
                plot_ax.plot(x_vals, y_vals, color="black", linewidth=1.5)

                y_pred = intercept + slope * is_perf
                ss_res = np.sum((oos_perf - y_pred) ** 2)
                ss_tot = np.sum((oos_perf - np.mean(oos_perf)) ** 2)
                r2 = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
            else:
                slope = 0.0
                intercept = float(oos_perf[0])
                r2 = 0.0

            adj_r2 = _adjusted_r2(r2, len(is_perf), 1)
            plot_ax.text(
                0.02,
                0.95,
                f"[SR OOS] = {intercept:.2f} + {slope:.2f}*[SR IS] + err | adjR2 = {adj_r2:.2f}",
                transform=plot_ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            prob_negative = float(np.mean(oos_perf < 0))
            plot_ax.text(
                0.02,
                0.05,
                f"Prob[SR OOS < 0] = {prob_negative:.2f}",
                transform=plot_ax.transAxes,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plot_ax.set_title("OOS Perf. Degradation")
        plot_ax.set_xlabel("SR IS")
        plot_ax.set_ylabel("SR OOS")
        plot_ax.grid(True)

        _finalize_plot(fig, save_path)

    def plot_stochastic_dominance(
        self,
        analysis_result: ModelAnalysisResult,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Plot second-order stochastic dominance diagnostics."""

        plt.style.use(self.config.plot_style)
        fig, plot_ax = _get_ax(ax, self.config.figure_size)

        is_perf, oos_perf = extract_best_is_performances(analysis_result)
        if is_perf.size == 0:
            plot_ax.text(0.5, 0.5, "No performance data", ha="center", va="center")
            _finalize_plot(fig, save_path)
            return

        x_values = np.sort(np.unique(np.concatenate([is_perf, oos_perf])))
        cdf_is = np.searchsorted(np.sort(is_perf), x_values, side="right") / is_perf.size
        cdf_oos = np.searchsorted(np.sort(oos_perf), x_values, side="right") / oos_perf.size

        plot_ax.plot(x_values, cdf_is, color="gray", label="optimized")
        plot_ax.plot(x_values, cdf_oos, color="blue", label="non-optimized")

        sd2 = np.cumsum(cdf_is - cdf_oos)
        ax2 = plot_ax.twinx()
        ax2.plot(x_values, sd2, color="red", label="SD2")
        ax2.set_ylabel("2nd Order Stoch. Dominance")
        ax2.set_ylim(min(sd2.min(), 0), 0)

        plot_ax.set_title("OOS Cumul. Dist.")
        plot_ax.set_xlabel("SR optimized vs. non-optimized")
        plot_ax.set_ylabel("Frequency")
        plot_ax.set_ylim(0, 1)
        plot_ax.legend(loc="upper left")
        plot_ax.grid(True)

        plot_ax.text(
            0.98,
            0.95,
            f"SD2 = {sd2[-1]:.2f}",
            transform=plot_ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        _finalize_plot(fig, save_path)

    def generate_report(self, analysis_result: ModelAnalysisResult, pbo_result: PBOResult) -> str:
        """Generate a text report for PBO analysis."""

        report = (
            "Probability of Backtest Overfitting (PBO) Report\n"
            f"Metric: {pbo_result.metric_used}\n"
            f"PBO: {pbo_result.pbo:.4f} ({pbo_result.interpretation})\n"
            f"Confidence Interval ({self.config.confidence_level:.0%}): "
            f"[{pbo_result.pbo_lower_ci:.4f}, {pbo_result.pbo_upper_ci:.4f}]\n"
            f"Performance Degradation (median IS - median OOS): {pbo_result.performance_degradation:.4f}\n"
            f"SD2 Statistic: {pbo_result.sd2_statistic:.4f}\n"
            f"Combinations: {pbo_result.n_combinations}\n"
            f"Strategies: {pbo_result.n_strategies}\n"
        )

        if pbo_result.pbo < 0.3:
            report += "Recommendation: Strategy set appears robust.\n"
        elif pbo_result.pbo < 0.5:
            report += "Recommendation: Investigate and validate with additional tests.\n"
        else:
            report += "Recommendation: High overfitting risk; consider reducing model complexity.\n"

        return report


def interpret_pbo(pbo: float) -> str:
    """Interpret PBO into qualitative categories."""

    if pbo < 0.3:
        return "Likely Robust - Low probability of overfitting"
    if pbo < 0.5:
        return "Questionable - Moderate overfitting concerns"
    return "Likely Overfit - High probability strategies are curve-fit"


def extract_best_is_performances(analysis_result: ModelAnalysisResult) -> Tuple[np.ndarray, np.ndarray]:
    """Extract IS and OOS performance of the best IS strategy per combination."""

    is_perf = []
    oos_perf = []
    for result in analysis_result.combination_results:
        idx = result.n_star
        is_perf.append(result.is_performance[idx])
        oos_perf.append(result.oos_performance[idx])
    return np.array(is_perf, dtype=float), np.array(oos_perf, dtype=float)


def _adjusted_r2(r2: float, n_samples: int, n_params: int) -> float:
    if n_samples <= n_params + 1:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_params - 1)


def _calculate_skewness(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    mean = np.mean(values)
    std = np.std(values, ddof=0)
    if std == 0:
        return 0.0
    centered = values - mean
    return float(np.mean(centered**3) / (std**3))


def _get_ax(ax: Optional[plt.Axes], figure_size: Tuple[int, int]) -> Tuple[Optional[plt.Figure], plt.Axes]:
    if ax is None:
        fig, plot_ax = plt.subplots(figsize=figure_size)
        return fig, plot_ax
    return None, ax


def _finalize_plot(fig: Optional[plt.Figure], save_path: Optional[str]) -> None:
    if fig is None:
        return
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    n_combinations = 200
    n_strategies = 10

    combination_results = []
    for i in range(n_combinations):
        is_ranks = np.arange(1, n_strategies + 1)
        n_star = 0

        if i < int(n_combinations * 0.65):
            oos_rank_best = rng.integers(1, (n_strategies // 2) + 1)
        else:
            oos_rank_best = rng.integers((n_strategies // 2) + 1, n_strategies + 1)

        oos_ranks = np.arange(1, n_strategies + 1)
        swap_idx = int(oos_rank_best - 1)
        oos_ranks[0], oos_ranks[swap_idx] = oos_ranks[swap_idx], oos_ranks[0]

        is_perf = -is_ranks.astype(float) + rng.normal(scale=0.01, size=n_strategies)
        oos_perf = -oos_ranks.astype(float) + rng.normal(scale=0.01, size=n_strategies)

        relative_rank = oos_rank_best / (n_strategies + 1)
        logit = math.log(relative_rank / (1 - relative_rank))
        combination_results.append(
            CombinationResult(
                combination_id=i,
                is_performance=is_perf,
                oos_performance=oos_perf,
                is_ranks=is_ranks,
                oos_ranks=oos_ranks,
                n_star=n_star,
                oos_rank_of_best_is=oos_rank_best,
                relative_rank=relative_rank,
                logit=logit,
            )
        )

    logits = np.array([result.logit for result in combination_results], dtype=float)

    analysis_result = ModelAnalysisResult(
        combination_results=combination_results,
        logits=logits,
        logit_distribution={},
        n_strategies=n_strategies,
        n_combinations=n_combinations,
        metric_used=PerformanceMetric.SHARPE_RATIO,
    )

    calculator = PBOCalculator(PBOConfig())
    pbo_result = calculator.calculate(analysis_result)

    print(calculator.generate_report(analysis_result, pbo_result))

    calculator.plot_all_diagnostics(analysis_result, pbo_result)
    calculator.plot_logit_distribution(logits, pbo_result.pbo, save_path="logit_hist.png")
    calculator.plot_performance_degradation_scatter(
        analysis_result, save_path="performance_degradation_scatter.png"
    )
    calculator.plot_stochastic_dominance(analysis_result, save_path="stochastic_dominance.png")
