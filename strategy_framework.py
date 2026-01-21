"""Reusable strategy framework for PBO analysis."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting_engine import BacktestConfig, BacktestEngine, BacktestResult


logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, backtest_config: Optional[BacktestConfig] = None) -> None:
        self.backtest_config = backtest_config or BacktestConfig()

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame, **params: Any) -> pd.Series:
        """Return signals as a pandas Series: 1 (long), -1 (short), 0 (neutral)."""

    @abstractmethod
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """Return list of parameter dictionaries to test."""

    def run_backtest_suite(
        self,
        prices: pd.DataFrame | pd.Series,
        price_col: str = "close",
        trim_to_divisible: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run backtests across the parameter grid and return M matrix with metadata.

        M matrix has shape (T x N), where T is number of observations and N is
        number of parameter combinations.
        """

        prices_df = self._normalize_prices(prices, price_col)

        original_start = prices_df.index.min()
        original_end = prices_df.index.max()

        if trim_to_divisible is not None:
            if trim_to_divisible <= 0:
                raise ValueError("trim_to_divisible must be positive")
            prices_df = self._trim_to_divisible(prices_df, trim_to_divisible)

        if prices_df.empty:
            raise ValueError("Price data is empty after trimming")

        grid = self.get_parameter_grid()
        if not grid:
            raise ValueError("Parameter grid is empty")

        engine = BacktestEngine(self.backtest_config)
        results: List[BacktestResult] = []

        for idx, params in enumerate(grid, start=1):
            if idx % 10 == 0:
                logger.info("Processed %s/%s backtests", idx, len(grid))

            result = engine.run(
                price_data=prices_df,
                strategy_func=self.generate_signals,
                strategy_params=params,
                price_col=price_col,
            )
            results.append(result)

        M_matrix = self._build_m_matrix(results)

        if np.isnan(M_matrix).any():
            raise ValueError("M matrix contains NaNs")

        metadata = {
            "strategy_name": self.__class__.__name__,
            "n_strategies": int(M_matrix.shape[1]),
            "n_observations": int(M_matrix.shape[0]),
            "date_range": {
                "start": prices_df.index.min().strftime("%Y-%m-%d"),
                "end": prices_df.index.max().strftime("%Y-%m-%d"),
            },
            "original_date_range": {
                "start": original_start.strftime("%Y-%m-%d"),
                "end": original_end.strftime("%Y-%m-%d"),
            },
            "parameter_grid": grid,
            "backtest_config": asdict(self.backtest_config),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        return M_matrix, metadata

    def save_for_pbo_analysis(
        self, M_matrix: np.ndarray, metadata: Dict[str, Any], output_dir: str = "data"
    ) -> None:
        """Save M matrix and metadata for PBO analysis."""

        os.makedirs(output_dir, exist_ok=True)
        matrix_path = os.path.join(output_dir, "M_matrix.npy")
        metadata_path = os.path.join(output_dir, "M_metadata.json")

        np.save(matrix_path, M_matrix)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved M matrix to {matrix_path}")
        print(f"Saved metadata to {metadata_path}")

    def _build_m_matrix(self, results: List[BacktestResult]) -> np.ndarray:
        """Extract pnl_series from each result and build M matrix."""

        if not results:
            raise ValueError("No backtest results to build M matrix")

        lengths = [len(result.pnl_series) for result in results]
        if len(set(lengths)) != 1:
            raise ValueError("Backtest results have mismatched lengths")

        columns = [result.pnl_series.to_numpy() for result in results]
        return np.column_stack(columns)

    def _normalize_prices(self, prices: pd.DataFrame | pd.Series, price_col: str) -> pd.DataFrame:
        if isinstance(prices, pd.Series):
            df = prices.to_frame(name=price_col)
        else:
            df = prices.copy()
            if price_col not in df.columns:
                if df.shape[1] == 1:
                    df.columns = [price_col]
                else:
                    raise ValueError(f"price_col '{price_col}' not found in prices")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df.sort_index()

    def _trim_to_divisible(self, df: pd.DataFrame, divisor: int) -> pd.DataFrame:
        total_len = len(df)
        remainder = total_len % divisor
        if remainder == 0:
            return df
        return df.iloc[:-remainder]




# ------ EXAMPLE STRATEGY IMPLEMENTATIONS ------

class MovingAverageCrossover(Strategy):
    """Example: MA crossover with configurable parameter grid."""

    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        return [
            {"fast": 10, "slow": 30},
            {"fast": 10, "slow": 50},
            {"fast": 20, "slow": 50},
            {"fast": 20, "slow": 100},
            {"fast": 30, "slow": 100},
            {"fast": 50, "slow": 100},
            {"fast": 50, "slow": 200},
            {"fast": 10, "slow": 100},
            {"fast": 30, "slow": 200},
            {"fast": 20, "slow": 200},
        ]

    def generate_signals(self, prices: pd.DataFrame, **params: Any) -> pd.Series:
        fast = int(params["fast"])
        slow = int(params["slow"])
        price_series = prices["close"]

        fast_ma = price_series.rolling(window=fast, min_periods=1).mean()
        slow_ma = price_series.rolling(window=slow, min_periods=1).mean()

        signals = pd.Series(0, index=price_series.index, dtype=float)
        signals[fast_ma > slow_ma] = 1.0
        signals[fast_ma < slow_ma] = -1.0
        return signals


class MeanReversion(Strategy):
    """Example: Mean reversion strategy with Z-score entry."""

    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        return [
            {"lookback": 20, "z_threshold": 2.0},
            {"lookback": 20, "z_threshold": 1.5},
            {"lookback": 50, "z_threshold": 2.0},
            {"lookback": 50, "z_threshold": 1.5},
            {"lookback": 100, "z_threshold": 2.0},
            {"lookback": 100, "z_threshold": 1.5},
        ]

    def generate_signals(self, prices: pd.DataFrame, **params: Any) -> pd.Series:
        lookback = int(params["lookback"])
        z_threshold = float(params["z_threshold"])
        price_series = prices["close"]

        rolling_mean = price_series.rolling(window=lookback, min_periods=1).mean()
        rolling_std = price_series.rolling(window=lookback, min_periods=1).std(ddof=0)
        z_score = (price_series - rolling_mean) / rolling_std.replace(0, np.nan)

        signals = pd.Series(0, index=price_series.index, dtype=float)
        signals[z_score > z_threshold] = -1.0
        signals[z_score < -z_threshold] = 1.0
        signals[(z_score <= z_threshold) & (z_score >= -z_threshold)] = 0.0
        return signals


# ------ STRATEGY EXECUTION ------

def quick_pbo_analysis(
    strategy: Strategy,
    prices: pd.DataFrame | pd.Series,
    S: int = 16,
    output_dir: str = "data",
) -> None:
    """Run a quick workflow to generate M matrix and metadata for PBO analysis."""

    if S % 2 != 0:
        raise ValueError("S must be even")

    M_matrix, metadata = strategy.run_backtest_suite(prices, trim_to_divisible=S)
    metadata["S"] = int(S)
    strategy.save_for_pbo_analysis(M_matrix, metadata, output_dir=output_dir)

    print("Data saved. Run PBO analysis with: python3 run_pbo_analysis.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    strategy = MovingAverageCrossover()
    quick_pbo_analysis(strategy, prices["SPY"], S=16)
