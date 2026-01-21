"""Data retrieval module for quantitative trading backtests."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLB",
    "XLU",
    "VNQ", # XLRE launched in 2015, too late for trials
    "VOX", # XLC launched in 2018, too late for trials
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "USO",
    "^VIX", # yfinance data for VXX is spotty
]

DEFAULT_FRED_SERIES = ["DGS10", "VIXCLS", "DTWEXBGS"]
DEFAULT_START_DATE = "2010-01-01"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retrying API calls."""

    max_retries: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0


RETRY_CONFIG = RetryConfig()


class DataRetrievalError(RuntimeError):
    """Raised when data retrieval fails after retries."""



def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)



def _setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)



def _retry_call(func: Callable[[], pd.DataFrame], context: str) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_CONFIG.max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - explicit logging and retry handling
            last_error = exc
            logger.error("Attempt %s/%s failed for %s: %s", attempt, RETRY_CONFIG.max_retries, context, exc)
            if attempt < RETRY_CONFIG.max_retries:
                delay = RETRY_CONFIG.base_delay * (RETRY_CONFIG.backoff_factor ** (attempt - 1))
                time.sleep(delay)
    raise DataRetrievalError(f"Failed to retrieve {context} after {RETRY_CONFIG.max_retries} retries") from last_error



def fetch_price_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for the supplied tickers from Yahoo Finance."""

    ticker_list = list(tickers)

    def _download() -> pd.DataFrame:
        data = yf.download(
            tickers=ticker_list,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise ValueError("No data returned from yfinance")
        if "Adj Close" in data.columns:
            adj_close = data["Adj Close"]
        else:
            adj_close = data.xs("Adj Close", axis=1, level=0, drop_level=False)
            adj_close.columns = adj_close.columns.droplevel(0)
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=ticker_list[0])
        return adj_close

    return _retry_call(_download, "price data")



def fetch_macro_data(
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    series: Iterable[str] = DEFAULT_FRED_SERIES,
) -> pd.DataFrame:
    """Fetch FRED macro indicators."""

    series_list = list(series)

    def _download() -> pd.DataFrame:
        data = web.DataReader(series_list, "fred", start_date, end_date)
        if data.empty:
            raise ValueError("No data returned from FRED")
        
        data.index.name = 'Date'
        data = data.dropna(how='all')
        return data

    return _retry_call(_download, "FRED macro data")



def fetch_ff_factors(
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch Fama-French 3 factors (daily) from Ken French's data library."""

    def _download() -> pd.DataFrame:
        ff_data = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start_date, end_date)
        if not ff_data or 0 not in ff_data:
            raise ValueError("No data returned from Fama-French")
        factors = ff_data[0].copy()
        factors.index = pd.to_datetime(factors.index)
        factors.index.name = 'Date'

        return factors

    return _retry_call(_download, "Fama-French factors")



def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data for missing values.

    Forward-fills gaps up to 5 days; logs a warning if larger gaps are found.
    """

    if df.empty:
        raise ValueError("DataFrame is empty")

    missing_counts = df.isna().sum()
    if missing_counts.any():
        logger.info("Missing values detected, attempting forward-fill")

    ffilled = df.ffill(limit=5)
    remaining = ffilled.isna()
    if remaining.any().any():
        columns_with_gaps = remaining.any()[remaining.any()].index.tolist()
        logger.warning("Missing data exceeds 5-day forward fill for columns: %s", columns_with_gaps)

    return ffilled



def save_data(df: pd.DataFrame, filename: str) -> str:
    """Save DataFrame to the data directory as CSV."""

    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    logger.info("Saved data to %s", path)
    return path



def load_data(filename: str) -> pd.DataFrame:
    """Load DataFrame from the data directory."""

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)



def update_data(
    filename: str,
    tickers: Iterable[str] = DEFAULT_TICKERS,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Update a saved price dataset by appending new data from Yahoo Finance."""

    existing = load_data(filename)
    if existing.empty:
        raise ValueError("Existing data is empty")

    last_date = existing.index.max().strftime("%Y-%m-%d")
    new_data = fetch_price_data(tickers, last_date, end_date)

    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    save_data(combined, filename)
    return combined



def _fetch_all_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch price, macro, and factor datasets in one call."""

    prices = fetch_price_data(tickers, start_date, end_date)
    macro = fetch_macro_data(start_date, end_date)
    factors = fetch_ff_factors(start_date, end_date)
    return prices, macro, factors


if __name__ == "__main__":
    _setup_logging()
    _ensure_data_dir()

    price_data, macro_data, ff_factors = _fetch_all_data(
        DEFAULT_TICKERS,
        DEFAULT_START_DATE,
        None,
    )

    price_data = validate_data(price_data)
    macro_data = validate_data(macro_data)
    ff_factors = validate_data(ff_factors)

    save_data(price_data, "prices.csv")
    save_data(macro_data, "macro.csv")
    save_data(ff_factors, "ff_factors.csv")
