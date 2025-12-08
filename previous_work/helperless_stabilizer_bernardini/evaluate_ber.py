import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add current directory to Python path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from formulas import delta_from_K_eta, eta_from_K_delta, eta_expr_from_K_delta, delta_from_K_maxvar_eta
from common.data_reading_utils import get_files, read_readouts, ReadoutList

@dataclass
class BerevalResult:
    K: int
    eta: float
    delta: float
    accepted_fraction: float
    heldout_ber_mean: float
    heldout_ber_min: float
    heldout_ber_max: float
    heldout_ber_variance: float
    heldout_rows_used: int


@dataclass
class BerevalTargetResult:
    K: int
    target_ber: float
    eta_selected: float | None
    delta: float | None
    accepted_fraction: float | None
    heldout_ber_mean: float | None
    heldout_ber_variance: float | None
    iters: int


def build_bit_matrix(readouts: ReadoutList) -> np.ndarray:
    """
    Stack all reads into a (R, B) uint8 matrix once per chip for speed.
    """
    return np.vstack([r.data.astype(np.uint8) for r in readouts])


def evaluate_ber_generalized(
    bit_matrix: np.ndarray,
    enrollment_reads: int,
    threshold: float,
    heldout_start: Optional[int] = None
) -> Tuple[float, float, float, float, float, int]:
    """
    Generalized BER evaluation function.
    
    Args:
        bit_matrix: (R, B) matrix of all reads
        enrollment_reads: Number of reads to use for enrollment (M)
        threshold: Threshold for acceptance mask (delta or D)
        heldout_start: Start index for heldout evaluation (default: enrollment_reads)
    
    Returns:
        (accepted_fraction, ber_mean, ber_min, ber_max, ber_variance, heldout_rows_used)
    """
    total_reads = bit_matrix.shape[0]
    M = int(enrollment_reads)
    if M <= 0 or M > total_reads:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0

    # Create acceptance mask and majority reference
    accept_mask = mask_from_threshold(bit_matrix, M, threshold)
    majority = majority_from_matrix(bit_matrix, M)
    
    # Determine heldout start
    if heldout_start is None:
        heldout_start = M
    if heldout_start >= total_reads:
        return accept_mask.mean().item(), float('nan'), float('nan'), float('nan'), float('nan'), 0

    # Get accepted cell indices
    accepted_idx = np.where(accept_mask)[0]
    if accepted_idx.size == 0:
        return 0.0, float('nan'), float('nan'), float('nan'), float('nan'), total_reads - heldout_start

    # Compute BER on heldout reads
    ref = majority[accepted_idx].astype(np.uint8)
    heldout = bit_matrix[heldout_start:, accepted_idx]
    flips_mat = ref ^ heldout
    if flips_mat.size == 0:
        return accept_mask.mean().item(), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    ber_per_read = flips_mat.mean(axis=1).astype(np.float64)
    return (
        float(accept_mask.mean()),
        float(np.nanmean(ber_per_read)),
        float(np.nanmin(ber_per_read)),
        float(np.nanmax(ber_per_read)),
        float(np.nanvar(ber_per_read)),
        int(heldout.shape[0]),
    )


def evaluate_heldout_ber(bit_matrix: np.ndarray, majority: np.ndarray, accept_mask: np.ndarray, start_idx: int) -> Tuple[float, float, float]:
    """
    Compute BER across held-out reads (from start_idx .. end) on accepted cells only.
    Returns mean/min/max BER across held-out reads.
    """
    total_reads = bit_matrix.shape[0]
    if start_idx >= total_reads:
        return float('nan'), float('nan'), float('nan')
    accepted_idx = np.where(accept_mask)[0]
    if accepted_idx.size == 0:
        return float('nan'), float('nan'), float('nan')

    ref = majority.astype(np.uint8)
    heldout = bit_matrix[start_idx:]
    flips_mat = ref[accepted_idx] ^ heldout[:, accepted_idx]
    if flips_mat.size == 0:
        return float('nan'), float('nan'), float('nan')
    ber_arr = flips_mat.mean(axis=1).astype(float)
    return float(np.nanmean(ber_arr)), float(np.nanmin(ber_arr)), float(np.nanmax(ber_arr))


def eta_from_exponent(exp: int | float) -> float:
    """
    Build eta as 1 - 10^{-exp} with high precision to avoid rounding to 1.0.
    """
    exp64 = np.float64(exp)
    term = np.power(np.float64(10.0), -exp64, dtype=np.float64)
    eta64 = np.float64(1.0) - term
    return float(eta64)  # Keep as float for compatibility but maintain precision


def evaluate_ber_for_grid(k_values: List[int], eta_values: List[float], chip_ids: Optional[List[str]] = None) -> Dict[str, List[BerevalResult]]:
    files = get_files()
    results: Dict[str, List[BerevalResult]] = {}
    selected = list(files.keys()) if chip_ids is None else [c for c in chip_ids if c in files]
    for chip_id in selected:
        file_list = files[chip_id]
        readouts = read_readouts(file_list)
        bit_matrix = build_bit_matrix(readouts)
        chip_results: List[BerevalResult] = []
        for K in k_values:
            K = int(K)
            for eta in eta_values:
                delta = float(delta_from_K_eta(int(K), float(eta)))
                accepted_fraction, ber_mean, ber_min, ber_max, ber_variance, heldout_rows = evaluate_ber_generalized(bit_matrix, K, delta, K)
                chip_results.append(BerevalResult(
                    K=int(K), eta=float(eta), delta=delta,
                    accepted_fraction=accepted_fraction,
                    heldout_ber_mean=float(ber_mean),
                    heldout_ber_min=float(ber_min),
                    heldout_ber_max=float(ber_max),
                    heldout_ber_variance=float(ber_variance),
                    heldout_rows_used=int(heldout_rows),
                ))
        results[chip_id] = chip_results
    return results


def save_results(results: Dict[str, List[BerevalResult]], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'ber_eval_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            chip: [r.__dict__ for r in rows]
            for chip, rows in results.items()
        }, f, indent=2)

    csv_path = output_dir / 'ber_eval_results.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('chip_id,K,eta,delta,accepted_fraction,heldout_ber_mean,heldout_ber_min,heldout_ber_max,heldout_ber_variance,heldout_rows_used\n')
        for chip, rows in results.items():
            for r in rows:
                f.write(f"{chip},{r.K},{r.eta:.15e},{r.delta},{r.accepted_fraction},{r.heldout_ber_mean},{r.heldout_ber_min},{r.heldout_ber_max},{r.heldout_ber_variance},{r.heldout_rows_used}\n")


def evaluate_ber_variance_based_delta(k_values: List[int], eta: float, chip_ids: Optional[List[str]] = None) -> Dict[str, List[BerevalResult]]:
    """
    Evaluate BER using variance-based delta calculation from equation 23.
    
    This is an ADDITIONAL evaluation method that computes delta using K, max_variance, and eta
    instead of the traditional delta_from_K_eta formula. It does NOT replace existing
    evaluation methods.
    
    Args:
        k_values: List of K values to evaluate
        eta: Confidence parameter (0 < eta < 1)
        chip_ids: Optional list of chip IDs to process (None for all)
    
    Returns:
        Dictionary mapping chip_id to list of BerevalResult objects
    """
    files = get_files()
    results: Dict[str, List[BerevalResult]] = {}
    selected = list(files.keys()) if chip_ids is None else [c for c in chip_ids if c in files]
    
    for chip_id in selected:
        file_list = files[chip_id]
        readouts = read_readouts(file_list)
        bit_matrix = build_bit_matrix(readouts)
        chip_results: List[BerevalResult] = []
        
        for K in k_values:
            K = int(K)
            
            # Compute maximum variance from the first K reads
            max_variance = compute_max_variance_from_matrix(bit_matrix, K)
            print(f"Max variance: {max_variance}")
            # Compute delta using variance-based formula (equation 23)
            delta = delta_from_K_maxvar_eta(K, max_variance, eta)
            print(f"Delta: {delta}")
            
            # Evaluate BER using the computed delta
            accepted_fraction, ber_mean, ber_min, ber_max, ber_variance, heldout_rows = evaluate_ber_generalized(bit_matrix, K, delta, K)
            
            chip_results.append(BerevalResult(
                K=K, eta=float(eta), delta=float(delta),
                accepted_fraction=accepted_fraction,
                heldout_ber_mean=float(ber_mean),
                heldout_ber_min=float(ber_min),
                heldout_ber_max=float(ber_max),
                heldout_ber_variance=float(ber_variance),
                heldout_rows_used=int(heldout_rows),
            ))
        
        results[chip_id] = chip_results
    
    return results


def evaluate_ber_fixed_delta(k_values: List[int], delta: float, chip_ids: Optional[List[str]] = None) -> Dict[str, List[BerevalResult]]:
    """
    Evaluate BER keeping the selection margin Δ fixed across K. For reporting,
    compute η(K, Δ) via eta_from_K_delta.
    """
    files = get_files()
    results: Dict[str, List[BerevalResult]] = {}
    selected = list(files.keys()) if chip_ids is None else [c for c in chip_ids if c in files]
    for chip_id in selected:
        file_list = files[chip_id]
        readouts = read_readouts(file_list)
        bit_matrix = build_bit_matrix(readouts)
        chip_results: List[BerevalResult] = []
        for K in k_values:
            K = int(K)
            accepted_fraction, ber_mean, ber_min, ber_max, ber_variance, heldout_rows = evaluate_ber_generalized(bit_matrix, K, delta, K)
            eta_est = eta_from_K_delta(int(K), float(delta))
            chip_results.append(BerevalResult(
                K=K, eta=float(np.float64(eta_est)), delta=float(delta),
                accepted_fraction=accepted_fraction,
                heldout_ber_mean=float(ber_mean),
                heldout_ber_min=float(ber_min),
                heldout_ber_max=float(ber_max),
                heldout_ber_variance=float(ber_variance),
                heldout_rows_used=int(heldout_rows),
            ))
        results[chip_id] = chip_results
    return results


def save_variance_based_results(results: Dict[str, List[BerevalResult]], output_dir: pathlib.Path, eta: float):
    """Save variance-based delta evaluation results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename suffix
    eta_str = f"{eta:.6f}".replace('.', 'p')
    
    json_path = output_dir / f'ber_eval_variance_based_eq23_eta{eta_str}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            chip: [r.__dict__ for r in rows]
            for chip, rows in results.items()
        }, f, indent=2)

    csv_path = output_dir / f'ber_eval_variance_based_eq23_eta{eta_str}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('chip_id,K,eta,delta_variance_based,accepted_fraction,heldout_ber_mean,heldout_ber_min,heldout_ber_max,heldout_ber_variance,heldout_rows_used\n')
        for chip, rows in results.items():
            for r in rows:
                f.write(f"{chip},{r.K},{r.eta:.15e},{r.delta},{r.accepted_fraction},{r.heldout_ber_mean},{r.heldout_ber_min},{r.heldout_ber_max},{r.heldout_ber_variance},{r.heldout_rows_used}\n")


def save_fixed_delta_results(results: Dict[str, List[BerevalResult]], output_dir: pathlib.Path, delta: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = ("%0.6f" % delta).replace('.', 'p')
    json_path = output_dir / f'ber_eval_fixed_delta_{tag}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            chip: [r.__dict__ for r in rows]
            for chip, rows in results.items()
        }, f, indent=2)

    csv_path = output_dir / f'ber_eval_fixed_delta_{tag}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('chip_id,K,eta_from_K_delta,eta_from_K_delta_expr,delta,accepted_fraction,heldout_ber_mean,heldout_ber_min,heldout_ber_max,heldout_ber_variance,heldout_rows_used\n')
        for chip, rows in results.items():
            for r in rows:
                expr = eta_expr_from_K_delta(int(r.K), float(delta))
                f.write(f"{chip},{r.K},{r.eta:.15e},{expr},{r.delta},{r.accepted_fraction},{r.heldout_ber_mean},{r.heldout_ber_min},{r.heldout_ber_max},{r.heldout_ber_variance},{r.heldout_rows_used}\n")


# --- New parametric BER evaluation functions ---

def reliabilities_from_matrix(bit_matrix: np.ndarray, M: int) -> np.ndarray:
    """Mean one-probability per cell over the first M enrollment reads."""
    M = int(M)
    if M <= 0 or M > bit_matrix.shape[0]:
        raise ValueError("Invalid M (enrollment reads).")
    return bit_matrix[:M].mean(axis=0).astype(np.float64)



def compute_max_variance_from_matrix(bit_matrix: np.ndarray, M: int) -> float:
    """
    Compute the maximum variance among cell values across M reads.
    
    For each cell, compute the variance as the average of (cell_value - mean)²
    across the first M reads, then return the maximum variance across all cells.
    
    This is the correct implementation for equation 23.
    
    Args:
        bit_matrix: (R, B) matrix of all reads
        M: Number of enrollment reads to use
    
    Returns:
        Maximum variance value across all cells
    """
    if M <= 0 or M > bit_matrix.shape[0]:
        raise ValueError("Invalid M (enrollment reads).")
    
    # Get the first M reads
    enrollment_data = bit_matrix[:M]  # Shape: (M, B)
    
    # Compute variance for each cell across M reads
    # For each cell: variance = mean((value - mean)²)
    cell_means = enrollment_data.mean(axis=0)  # Shape: (B,)
    
    # Compute (value - mean)² for each cell and reading
    squared_diffs = (enrollment_data - cell_means[np.newaxis, :]) ** 2  # Shape: (M, B)
    
    # Average across M readings for each cell
    cell_variances = squared_diffs.mean(axis=0)  # Shape: (B,)
    
    # Return the maximum variance across all cells
    return float(np.max(cell_variances))


def mask_from_threshold(bit_matrix: np.ndarray, M: int, threshold: float) -> np.ndarray:
    """
    Acceptance mask using enrollment reads M and threshold T (delta or D):
    accept if |p_hat - 0.5| >= T.
    """
    p_hat = reliabilities_from_matrix(bit_matrix, M)
    return (np.abs(p_hat - 0.5) >= np.float64(threshold))


def majority_from_matrix(bit_matrix: np.ndarray, M: int) -> np.ndarray:
    """
    Majority decision over first M reads: True if majority ones, else False.
    Ties at exactly 0.5 go to 0 by construction (>= 0.5 -> 1).
    """
    M = int(M)
    return (bit_matrix[:M].mean(axis=0) >= 0.5).astype(np.uint8)


def evaluate_regen_ber_with_threshold(
    bit_matrix: np.ndarray,
    N: int,
    threshold: float,
    heldout_start: Optional[int] = None
) -> Tuple[float, float, float, float, float, int]:
    """
    Compute regeneration BER using acceptance mask built with threshold (D) on N reads,
    then evaluate flips on remaining reads (default: from N to end).
    Returns: (accepted_fraction, ber_mean, ber_min, ber_max, ber_variance, used_heldout_rows).
    """
    return evaluate_ber_generalized(bit_matrix, N, threshold, heldout_start)


def evaluate_enrollment_mask_ber(
    bit_matrix: np.ndarray,
    base_K: int,
    base_delta: float,
    block_N: int,
    block_threshold: float,
    start_after_base: Optional[int] = None
) -> Tuple[float, int, int]:
    """
    Compare the acceptance mask built with (base_delta, base_K) to masks built from
    subsequent non-overlapping blocks of length block_N using (block_threshold, block_N).
    For each block, compute the fraction of cells where the masks differ; return the average
    over blocks. This is the 'enrollment BER' between masks.

    Returns: (avg_mask_disagreement, blocks_used, cells_per_mask)
    """
    total_reads, num_cells = bit_matrix.shape
    base_K = int(base_K); block_N = int(block_N)

    base_mask = mask_from_threshold(bit_matrix, base_K, base_delta)

    if start_after_base is None:
        start = base_K
    else:
        start = int(start_after_base)

    if start >= total_reads or block_N <= 0:
        return float('nan'), 0, int(num_cells)

    blocks = (total_reads - start) // block_N
    if blocks <= 0:
        return float('nan'), 0, int(num_cells)

    disagreements = []
    for b in range(blocks):
        s = start + b * block_N
        e = s + block_N
        sub = bit_matrix[s:e]
        # mask from exactly these block_N reads
        p_hat_block = sub.mean(axis=0).astype(np.float64)
        block_mask = (np.abs(p_hat_block - 0.5) >= np.float64(block_threshold))
        # disagreement rate over all cells (mask vs mask)
        diff = (base_mask ^ block_mask).astype(np.uint8)
        disagreements.append(diff.mean().item())

    return float(np.mean(disagreements)), int(blocks), int(num_cells)


def find_eta_for_target_ber(bit_matrix: np.ndarray,
                            K: int,
                            target_ber: float,
                            eta_low_bound: float = 0.99,
                            eta_high_bound: float = 0.999999,
                            max_iters: int = 24,
                            tol: float = 1e-4) -> tuple[float | None, float | None, float | None, float | None, float | None, int]:
    """
    Bisection over eta to achieve held-out BER <= target_ber.
    Returns (eta_selected, delta, accepted_fraction, ber_mean, ber_variance, iters). None if not achievable.
    """
    low = float(eta_low_bound)
    high = float(eta_high_bound)
    best_eta = None
    best_tuple = (None, None, None, None, None)

    # Evaluate at bounds to ensure the target is bracketed
    def eval_eta(e: float) -> tuple[float, float, float, float]:
        d = float(delta_from_K_eta(int(K), e))
        acc_frac, bmean, _, _, bvar, _ = evaluate_ber_generalized(bit_matrix, K, d, K)
        return d, acc_frac, bmean, bvar

    d_low, acc_low, ber_low, var_low = eval_eta(low)
    _, _, ber_high, _ = eval_eta(high)

    # If even the strictest (high) can't meet target, return None
    if not np.isnan(ber_high) and ber_high > target_ber + tol:
        return None, None, None, None, None, 1
    # If the loosest (low) already meets target, pick low
    if not np.isnan(ber_low) and ber_low <= target_ber + tol:
        return float(low), float(d_low), float(acc_low), float(ber_low), float(var_low), 1

    for it in range(max_iters):
        mid = (low + high) / 2.0
        delta = float(delta_from_K_eta(int(K), mid))
        accepted_fraction, ber_mean, _, _, ber_variance, _ = evaluate_ber_generalized(bit_matrix, K, delta, K)
        if np.isnan(ber_mean):
            # If no accepted cells, increase acceptance by reducing eta (smaller delta)
            high = mid
            continue
        # Larger eta -> larger delta -> typically lower BER
        if ber_mean <= target_ber + tol:
            best_eta = mid
            best_tuple = (delta, accepted_fraction, ber_mean, ber_variance)
            # try to lower eta further to accept more cells while meeting target
            high = mid
        else:
            # need stricter delta -> increase eta
            low = mid
        if abs(high - low) < 1e-12:
            break
    if best_eta is None:
        return None, None, None, None, None, it + 1
    return float(best_eta), float(best_tuple[0]), float(best_tuple[1]), float(best_tuple[2]), float(best_tuple[3]), it + 1


def evaluate_ber_for_target(k_values: List[int], target_ber: float,
                            eta_low_bound: float = 0.99, eta_high_bound: float = 0.999999,
                            chip_ids: Optional[List[str]] = None) -> Dict[str, List[BerevalTargetResult]]:
    files = get_files()
    results: Dict[str, List[BerevalTargetResult]] = {}
    selected = list(files.keys()) if chip_ids is None else [c for c in chip_ids if c in files]
    for chip_id in selected:
        file_list = files[chip_id]
        readouts = read_readouts(file_list)
        bit_matrix = build_bit_matrix(readouts)
        chip_rows: List[BerevalTargetResult] = []
        for K in k_values:
            K = int(K)
            eta_sel, delta, acc_frac, ber_mean, ber_variance, iters = find_eta_for_target_ber(
                bit_matrix, int(K), float(target_ber), eta_low_bound, eta_high_bound
            )
            chip_rows.append(BerevalTargetResult(
                K=int(K), target_ber=float(target_ber), eta_selected=None if eta_sel is None else float(eta_sel),
                delta=None if delta is None else float(delta),
                accepted_fraction=None if acc_frac is None else float(acc_frac),
                heldout_ber_mean=None if ber_mean is None else float(ber_mean),
                heldout_ber_variance=None if ber_variance is None else float(ber_variance),
                iters=int(iters),
            ))
        results[chip_id] = chip_rows
    return results


def save_target_results(results: Dict[str, List[BerevalTargetResult]], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'ber_eval_target_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            chip: [r.__dict__ for r in rows]
            for chip, rows in results.items()
        }, f, indent=2)

    csv_path = output_dir / 'ber_eval_target_results.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('chip_id,K,target_ber,eta_selected,delta,accepted_fraction,heldout_ber_mean,heldout_ber_variance,iters\n')
        for chip, rows in results.items():
            for r in rows:
                eta_str = f"{r.eta_selected:.15e}" if r.eta_selected is not None else "None"
                var_str = f"{r.heldout_ber_variance:.15e}" if r.heldout_ber_variance is not None else "None"
                f.write(f"{chip},{r.K},{r.target_ber},{eta_str},{r.delta},{r.accepted_fraction},{r.heldout_ber_mean},{var_str},{r.iters}\n")


if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).parent
    out_dir = base_dir / 'results'

    # Use all 55k cells and 1000 reads implicitly via data utils.
    # Configure K, eta grid (for grid mode), target BER, and fixed-delta
    k_grid = [68, 100, 200, 500]
    # Provide eta as exponents e where eta = 1 - 10^{-e}
    # eta_grid_exponents = [6, 9, 12]
    eta_grid_exponents = [6]
    eta_grid = [eta_from_exponent(e) for e in eta_grid_exponents]


    # Optionally restrict to specific chips
    chips_filter: Optional[List[str]] = ['L45']  # or None, e.g., ['M2', 'M49']
    res = evaluate_ber_for_grid(k_grid, eta_grid, chip_ids=chips_filter)
    save_results(res, out_dir)

    # Example: target BER search across eta per K
    target = 1e-4
    # Use an eta bracket consistent with the grid (or override via exponents)
    eta_low_exp = min(eta_grid_exponents)
    eta_high_exp = max(eta_grid_exponents)
    eta_low_val = eta_from_exponent(eta_low_exp)
    eta_high_val = eta_from_exponent(eta_high_exp)
    res_target = evaluate_ber_for_target(k_grid, target,
                                         eta_low_bound=float(eta_low_val),
                                         eta_high_bound=float(eta_high_val),
                                         chip_ids=chips_filter)
    save_target_results(res_target, out_dir)
    
    # Example: variance-based delta calculation (equation 23)
    eta_variance = eta_from_exponent(6)  # Use eta = 1 - 10^-6
    res_variance_eq23 = evaluate_ber_variance_based_delta(k_grid, eta_variance, chip_ids=chips_filter)
    save_variance_based_results(res_variance_eq23, out_dir, eta_variance)
    
    print("Variance-based delta evaluation completed!")
    print(f"Results saved for eta = {eta_variance:.15e}")
    print("Files:")
    print(f"  - Equation 23: ber_eval_variance_based_eq23_eta{eta_variance:.6f}.csv")

    # Fixed-delta evaluation across K (keeps Δ constant and derives η(K, Δ) for reporting)
    delta_fixed = 0.25 # 0.499
    res_fixed = evaluate_ber_fixed_delta(k_grid, delta_fixed, chip_ids=chips_filter)
    save_fixed_delta_results(res_fixed, out_dir, delta_fixed)


