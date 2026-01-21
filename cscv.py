"""Combinatorially Symmetric Cross-Validation (CSCV) partitioning utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CSCVConfig:
    """Configuration for CSCV partitioning."""

    S: int = 16


@dataclass(frozen=True)
class CSCVSplit:
    """Container for a single CSCV train/test split."""

    J_train: np.ndarray
    J_test: np.ndarray
    train_indices: List[int]
    test_indices: List[int]
    combination_id: int


class CSCVPartitioner:
    """Partition a P&L matrix into CSCV train/test splits."""

    def __init__(self, M: np.ndarray, config: CSCVConfig | None = None) -> None:
        self.config = config or CSCVConfig()
        self.M = M
        self._validate_M()
        self._submatrices = self._partition_into_submatrices()
        self._combinations = self._generate_combinations()

        total_combinations = len(self)
        if total_combinations > 50_000:
            logger.warning("CSCV combinations count is large (%s)", total_combinations)

    @property
    def n_strategies(self) -> int:
        """Return number of strategies (N)."""

        return self.M.shape[1]

    @property
    def n_observations(self) -> int:
        """Return number of observations (T)."""

        return self.M.shape[0]

    @property
    def n_submatrices(self) -> int:
        """Return number of submatrices (S)."""

        return self.config.S

    def _validate_M(self) -> None:
        if not isinstance(self.M, np.ndarray):
            raise ValueError("M must be a numpy array")
        if self.M.ndim != 2:
            raise ValueError("M must be a 2D numpy array")

        if np.isnan(self.M).any():
            raise ValueError("M contains NaNs; please clean the data before partitioning")

        T, N = self.M.shape
        if N < 2:
            raise ValueError("M must contain at least 2 strategies (N >= 2)")

        if self.config.S % 2 != 0:
            raise ValueError("S must be even")

        if T % self.config.S != 0:
            raise ValueError("S must evenly divide T")

    def _partition_into_submatrices(self) -> List[np.ndarray]:
        T = self.n_observations
        S = self.config.S
        rows_per_block = T // S
        submatrices = []
        for idx in range(S):
            start = idx * rows_per_block
            end = start + rows_per_block
            submatrices.append(self.M[start:end, :])
        return submatrices

    def _generate_combinations(self) -> List[Tuple[List[int], List[int]]]:
        S = self.config.S
        train_size = S // 2
        all_indices = list(range(S))

        combo_list: List[Tuple[List[int], List[int]]] = []
        for train_indices in combinations(all_indices, train_size):
            train_list = list(train_indices)
            test_list = [idx for idx in all_indices if idx not in train_list]
            combo_list.append((train_list, test_list))
        return combo_list

    def get_split(self, combination_id: int) -> CSCVSplit:
        if combination_id < 0 or combination_id >= len(self):
            raise IndexError("combination_id out of range")

        train_indices, test_indices = self._combinations[combination_id]
        train_blocks = [self._submatrices[idx] for idx in train_indices]
        test_blocks = [self._submatrices[idx] for idx in test_indices]

        J_train = np.vstack(train_blocks)
        J_test = np.vstack(test_blocks)

        return CSCVSplit(
            J_train=J_train,
            J_test=J_test,
            train_indices=train_indices,
            test_indices=test_indices,
            combination_id=combination_id,
        )

    def get_all_splits(self) -> List[CSCVSplit]:
        return [self.get_split(idx) for idx in range(len(self))]

    def __iter__(self) -> Iterator[CSCVSplit]:
        for idx in range(len(self)):
            yield self.get_split(idx)

    def __len__(self) -> int:
        return self.get_num_combinations()

    def get_num_combinations(self) -> int:
        S = self.config.S
        return math.comb(S, S // 2)


def calculate_M_from_pnl_series(pnl_list: List[pd.Series]) -> np.ndarray:
    """Align a list of P&L series and return an (T x N) numpy array."""

    if len(pnl_list) < 2:
        raise ValueError("pnl_list must contain at least two series")

    aligned = pd.concat(pnl_list, axis=1, join="inner").sort_index()
    if aligned.empty:
        raise ValueError("No overlapping dates across series")

    filled = aligned.ffill(limit=5)
    if filled.isna().any().any():
        raise ValueError("Missing data exceeds 5-day forward-fill limit")

    return filled.to_numpy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    T = 1000
    N = 10
    S = 8
    rng = np.random.default_rng(42)
    M = rng.normal(size=(T, N))

    partitioner = CSCVPartitioner(M, CSCVConfig(S=S))
    print(f"Number of combinations: {partitioner.get_num_combinations()}")

    first_split = partitioner.get_split(0)
    print("First split train shape:", first_split.J_train.shape)
    print("First split test shape:", first_split.J_test.shape)

    for i, split in zip(range(3), partitioner):
        print(f"Split {i} train shape: {split.J_train.shape}, test shape: {split.J_test.shape}")
