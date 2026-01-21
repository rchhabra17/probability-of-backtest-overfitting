"""End-to-end PBO analysis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from analyze_models import AnalysisConfig, ModelAnalysisResult, ModelAnalyzer, PerformanceMetric
from cscv import CSCVConfig, CSCVPartitioner
from pbo import PBOCalculator, PBOConfig, PBOResult


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_inputs(data_dir: str) -> None:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    matrix_path = os.path.join(data_dir, "M_matrix.npy")
    metadata_path = os.path.join(data_dir, "M_metadata.json")
    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(f"M matrix file not found: {matrix_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")


def _parse_metric(metric: str) -> PerformanceMetric:
    try:
        return PerformanceMetric(metric.lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in PerformanceMetric)
        raise ValueError(f"Invalid metric '{metric}'. Valid options: {valid}") from exc


def _print_metadata_summary(metadata: dict) -> None:
    strategy_params = metadata.get("strategy_params", [])
    trimmed_range = metadata.get("trimmed_date_range", {})
    original_range = metadata.get("original_date_range", {})

    logger.info("Metadata summary:")
    logger.info("Strategies: %s", len(strategy_params))
    logger.info("Original range: %s to %s", original_range.get("start"), original_range.get("end"))
    logger.info("Trimmed range: %s to %s", trimmed_range.get("start"), trimmed_range.get("end"))
    logger.info("T: %s | N: %s | S: %s", metadata.get("T"), metadata.get("N"), metadata.get("S"))


def _save_pickle(obj: object, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _build_analysis_result(
    analyzer: ModelAnalyzer,
    partitioner: CSCVPartitioner,
    M: np.ndarray,
) -> ModelAnalysisResult:
    combination_results: List = []
    total = len(partitioner)

    for idx, split in enumerate(tqdm(partitioner, total=total, desc="Analyzing combinations")):
        if idx > 0 and idx % 1000 == 0:
            logger.info("Processed %s/%s combinations", idx, total)
        combination_results.append(analyzer._analyze_single_combination(split, M))

    logits = np.array([result.logit for result in combination_results], dtype=float)
    logit_distribution = analyzer._build_logit_distribution(logits)

    return ModelAnalysisResult(
        combination_results=combination_results,
        logits=logits,
        logit_distribution=logit_distribution,
        n_strategies=M.shape[1],
        n_combinations=len(partitioner),
        metric_used=analyzer.config.performance_metric,
    )


def run_pipeline(metric: PerformanceMetric, s_override: Optional[int], output_dir: str) -> None:
    start_time = time.time()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    _validate_inputs(data_dir)

    matrix_path = os.path.join(data_dir, "M_matrix.npy")
    metadata_path = os.path.join(data_dir, "M_metadata.json")

    M = np.load(matrix_path)
    metadata = _load_metadata(metadata_path)
    _print_metadata_summary(metadata)

    if M.ndim != 2:
        raise ValueError("M matrix must be 2D")

    T = int(metadata.get("T", M.shape[0]))
    N = int(metadata.get("N", M.shape[1]))
    S = int(metadata.get("S", 16))
    if s_override is not None:
        S = s_override

    if M.shape[0] != T or M.shape[1] != N:
        raise ValueError(f"M matrix shape {M.shape} does not match metadata T={T}, N={N}")

    os.makedirs(output_dir, exist_ok=True)

    partitioner = CSCVPartitioner(M, CSCVConfig(S=S))
    logger.info("Generated %s combinations with %s days per submatrix", len(partitioner), T // S)

    analyzer = ModelAnalyzer(AnalysisConfig(performance_metric=metric))

    analysis_result: Optional[ModelAnalysisResult] = None
    try:
        analysis_result = _build_analysis_result(analyzer, partitioner, M)
        analysis_path = os.path.join(output_dir, "analysis_result.pkl")
        _save_pickle(analysis_result, analysis_path)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving partial analysis results.")
        if analysis_result is not None:
            analysis_path = os.path.join(output_dir, "analysis_result_partial.pkl")
            _save_pickle(analysis_result, analysis_path)
        raise

    calculator = PBOCalculator(PBOConfig())
    pbo_result = calculator.calculate(analysis_result)

    pbo_path = os.path.join(output_dir, "pbo_result.pkl")
    _save_pickle(pbo_result, pbo_path)

    report = calculator.generate_report(analysis_result, pbo_result)
    print(report)

    diagnostics_path = os.path.join(output_dir, "pbo_diagnostics.png")
    calculator.plot_all_diagnostics(analysis_result, pbo_result, save_path=diagnostics_path)

    report_path = os.path.join(output_dir, "pbo_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    elapsed = time.time() - start_time
    trimmed_range = metadata.get("trimmed_date_range", {})

    logger.info("Summary:")
    logger.info("Date range: %s to %s (%s days)", trimmed_range.get("start"), trimmed_range.get("end"), T)
    logger.info("Strategies: %s", N)
    logger.info("CSCV combinations: %s", len(partitioner))
    logger.info("PBO: %.4f | %s", pbo_result.pbo, pbo_result.interpretation)
    logger.info("Performance degradation: %.4f", pbo_result.performance_degradation)
    logger.info("SD2 statistic: %.4f", pbo_result.sd2_statistic)
    logger.info("Saved analysis: %s", analysis_path)
    logger.info("Saved PBO result: %s", pbo_path)
    logger.info("Saved report: %s", report_path)
    logger.info("Saved diagnostics: %s", diagnostics_path)
    logger.info("Execution time: %.2f seconds", elapsed)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full PBO analysis pipeline")
    parser.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="Performance metric to rank strategies (default: sharpe_ratio)",
    )
    parser.add_argument("--s", type=int, default=None, help="Override S parameter from metadata")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Output directory for results (default: results/)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    _setup_logging()
    args = _parse_args(argv)

    try:
        metric = _parse_metric(args.metric)
        run_pipeline(metric, args.s, args.output_dir)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
