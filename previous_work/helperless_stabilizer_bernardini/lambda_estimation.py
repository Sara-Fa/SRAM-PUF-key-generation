import json
import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

from common.data_reading_utils import get_files, read_readouts, ReadoutList
from .formulas import p_unreliable_exact


@dataclass
class LambdaEstimate:
    lambda_window: float
    window_delta: float
    num_cells: int
    num_readouts: int
    validation: dict


def compute_cell_reliabilities(readouts: ReadoutList) -> np.ndarray:
    """
    Compute per-cell reliability x_i as the empirical fraction of the dominant
    value across K readings for each cell.
    """
    # Build matrix shape = (K, num_bits)
    K = len(readouts)
    bit_matrix = np.vstack([r.data for r in readouts])  # (K, B)
    # Majority per bit
    ones_count = np.sum(bit_matrix, axis=0)
    zeros_count = K - ones_count
    majority = (ones_count >= zeros_count).astype(np.uint8)
    # Reliability per bit = matches / K
    matches = np.sum(bit_matrix == majority, axis=0)
    reliabilities = matches.astype(np.float64) / float(K)
    return reliabilities


def estimate_lambda_window(reliabilities: np.ndarray, window_delta: float) -> float:
    mask = (reliabilities >= 0.5 - window_delta) & (reliabilities <= 0.5 + window_delta)
    p_unrel_emp = np.count_nonzero(mask) / reliabilities.size
    return p_unrel_emp / (2.0 * window_delta)


def validate_window(reliabilities: np.ndarray, lambda_hat: float, window_delta: float,
                    relative_tol: float = 0.02) -> dict:
    """
    A posteriori verification that Δ_window is small enough and approximation holds.
    - Compare empirical fraction in window to exact integral with λ̂
    - Compare exact integral to linear approximation 2Δ_window·λ̂
    - Check stability of λ̂ when halving Δ_window
    """
    # Empirical fraction in current window
    mask = (reliabilities >= 0.5 - window_delta) & (reliabilities <= 0.5 + window_delta)
    p_emp = np.count_nonzero(mask) / reliabilities.size

    # Exact probability under model with lambda_hat
    p_exact = float(p_unreliable_exact(window_delta, lambda_hat))
    p_approx = 2.0 * window_delta * lambda_hat

    rel_err_emp = float(abs(p_exact - p_emp) / p_exact) if p_exact > 0 else float('inf')
    rel_err_approx = float(abs(p_exact - p_approx) / p_exact) if p_exact > 0 else float('inf')

    # Stability check by halving window size
    half_delta = window_delta / 2.0
    lambda_half = estimate_lambda_window(reliabilities, half_delta)
    stability_rel_change = float(abs(lambda_half - lambda_hat) / lambda_hat) if lambda_hat > 0 else float('inf')
    
    # Add overall validation status
    all_checks_passed = (
        rel_err_emp <= relative_tol and 
        rel_err_approx <= relative_tol and 
        stability_rel_change <= relative_tol
    )

    return {
        'window_delta': float(window_delta),
        'p_empirical': float(p_emp),
        'p_exact': float(p_exact),
        'p_approx_linear': float(p_approx),
        'rel_err_empirical_vs_exact': rel_err_emp,
        'rel_err_approx_vs_exact': rel_err_approx,
        'lambda_half_window': float(lambda_half),
        'stability_rel_change': stability_rel_change,
        'empirical_within_tol': bool(rel_err_emp <= relative_tol),
        'approx_within_tol': bool(rel_err_approx <= relative_tol),
        'stability_within_tol': bool(stability_rel_change <= relative_tol),
        'relative_tol': float(relative_tol),
        'all_validation_checks_passed': bool(all_checks_passed),
        'validation_summary': 'PASS' if all_checks_passed else 'FAIL'
    }


def estimate_and_save_lambda(output_dir: pathlib.Path, window_delta: float = 0.02) -> Dict[str, LambdaEstimate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chip_to_files = get_files()
    results: Dict[str, LambdaEstimate] = {}

    for chip_id, files in chip_to_files.items():
        readouts = read_readouts(files)
        reliabilities = compute_cell_reliabilities(readouts)

        lam_window = estimate_lambda_window(reliabilities, window_delta)
        validation = validate_window(reliabilities, lam_window, window_delta)

        est = LambdaEstimate(
            lambda_window=float(lam_window),
            window_delta=float(window_delta),
            num_cells=int(reliabilities.size),
            num_readouts=int(len(readouts)),
            validation=validation,
        )
        results[chip_id] = est

    # Save JSON summary
    json_path = output_dir / 'lambda_estimates.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({k: est.__dict__ for k, est in results.items()}, f, indent=2)

    return results


if __name__ == '__main__':
    out_dir = pathlib.Path(__file__).parent / 'results'
    estimate_and_save_lambda(out_dir)


