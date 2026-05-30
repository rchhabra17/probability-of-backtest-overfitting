"""Modular backtesting engine for single-asset strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""

    initial_capital: float = 100000.0
    transaction_cost: float = 0.0005
    slippage: float = 0.0005
    position_size_method: str = "fixed"
    max_position_size: float = 1.0


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    pnl_series: pd.Series
    returns_series: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Strategy-agnostic backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        price_data: pd.DataFrame,
        strategy_func: Callable[..., pd.Series],
        strategy_params: Optional[Dict[str, object]] = None,
        price_col: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a backtest for a single asset.

        The strategy function must return signals in {-1, 0, 1}.
        """

        if price_data.empty:
            raise ValueError("price_data is empty")

        strategy_params = strategy_params or {}
        prices = self._extract_prices(price_data, price_col)

        signals = strategy_func(price_data, **strategy_params)
        positions = self._signals_to_positions(signals, prices.index)

        returns_series, pnl_series, equity_curve = self._calculate_returns(prices, positions)
        trades = self._generate_trades(prices, positions)
        metrics = self._calculate_metrics(returns_series, pnl_series, equity_curve, trades)

        return BacktestResult(
            pnl_series=pnl_series,
            returns_series=returns_series,
            positions=positions,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    def _extract_prices(self, price_data: pd.DataFrame, price_col: Optional[str]) -> pd.Series:
        if price_col and price_col in price_data.columns:
            prices = price_data[price_col]
        elif price_data.shape[1] == 1:
            prices = price_data.iloc[:, 0]
        else:
            raise ValueError("price_col must be provided when price_data has multiple columns")

        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        return prices.sort_index()

    def _signals_to_positions(self, signals: pd.Series, index: Iterable[pd.Timestamp]) -> pd.Series:
        if not isinstance(signals, pd.Series):
            raise ValueError("strategy function must return a pandas Series")

        positions = signals.reindex(index).ffill().fillna(0.0)
        if self.config.position_size_method == "fixed":
            positions = positions * self.config.max_position_size
        elif self.config.position_size_method == "percent":
            positions = positions * self.config.max_position_size
        else:
            logger.warning("Unknown position_size_method '%s'; defaulting to fixed", self.config.position_size_method)
            positions = positions * self.config.max_position_size

        positions = positions.clip(-self.config.max_position_size, self.config.max_position_size)
        return positions.astype(float)

    def _calculate_returns(
        self,
        prices: pd.Series,
        positions: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        price_returns = prices.pct_change().fillna(0.0)
        lagged_positions = positions.shift(1).fillna(0.0)

        turnover = positions.diff().abs().fillna(0.0)
        cost_rate = self.config.transaction_cost + self.config.slippage
        costs = turnover * cost_rate

        strategy_returns = (lagged_positions * price_returns) - costs
        equity_curve = (1.0 + strategy_returns).cumprod() * self.config.initial_capital
        pnl_series = equity_curve.diff().fillna(0.0)

        return strategy_returns, pnl_series, equity_curve

    def _generate_trades(self, prices: pd.Series, positions: pd.Series) -> pd.DataFrame:
        position_changes = positions.diff().fillna(positions)
        trade_mask = position_changes != 0

        trades = pd.DataFrame(
            {
                "date": positions.index[trade_mask],
                "position_change": position_changes[trade_mask].values,
                "direction": np.where(position_changes[trade_mask] > 0, "buy", "sell"),
                "size": position_changes[trade_mask].abs().values,
                "price": prices[trade_mask].values,
            }
        )
        return trades.reset_index(drop=True)

    def _calculate_metrics(
        self,
        returns_series: pd.Series,
        pnl_series: pd.Series,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        total_return = equity_curve.iloc[-1] / self.config.initial_capital - 1.0
        if len(returns_series) > 0:
            annual_return = (1.0 + total_return) ** (252.0 / len(returns_series)) - 1.0
        else:
            annual_return = 0.0

        volatility = returns_series.std(ddof=0) * np.sqrt(252)
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = returns_series.mean() * 252 / volatility

        running_max = equity_curve.cummax()
        drawdowns = equity_curve / running_max - 1.0
        max_drawdown = drawdowns.min()

        win_rate = 0.0
        if len(pnl_series) > 0:
            win_rate = (pnl_series > 0).mean()

        num_trades = float(len(trades))
        avg_trade_size = float(trades["size"].mean()) if len(trades) > 0 else 0.0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "num_trades": num_trades,
            "avg_trade_size": avg_trade_size,
        }

