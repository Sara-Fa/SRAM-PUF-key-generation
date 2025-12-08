"""Aggregation utilities for Bernardini experiment caches.

This module aggregates results saved by:
- GlobalBERProcessor via BERCacheManager (files: regenerate_ber_*.pkl)
- HelperDataComparator via ComparatorCacheManager (files: enroll_comparator_*.pkl)

It provides helpers to average over enrollment ranges and chips, and to
prepare data for plotting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pickle
import numpy as np
from pathlib import Path

from common.data_reading_utils import get_files, read_readouts, ReadoutList
from .global_ber_processor import GlobalBERProcessor
from .helper_data_comparator import HelperDataComparator


def _list_cache_files(cache_dir: Path, prefix: str, suffix: str) -> List[Path]:
    return sorted([p for p in cache_dir.iterdir() if p.name.startswith(prefix) and p.name.endswith(suffix)])


def aggregate_global_ber_over_chips(
    chip_ids: List[str],
    num_enroll_readings: int,
    cache_dir: Path = Path("previous_work/helperless_stabilizer_bernardini/experiments/cache"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate Global BER across chips.

    For each chip, loads BER cache files (regenerate_ber_...) for the specified
    num_enroll_readings. Converts counts to BER using that chip's heldout count
    (total_reads - num_enroll_readings). Averages BER over enrollment ranges
    (and over iterations if present), then averages over chips.

    Returns:
        thresholds: np.ndarray of unique thresholds (sorted ascending)
        mean_ber: np.ndarray of mean BER per threshold over chips
        std_ber: np.ndarray of std BER per threshold over chips
        mean_accept: np.ndarray of mean acceptance rate per threshold over chips
        std_accept: np.ndarray of std acceptance rate per threshold over chips
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Data files to determine heldout counts per chip
    all_files = get_files()

    # Collect per-chip maps: threshold -> scalar BER / acceptance
    per_chip_maps: List[Dict[float, float]] = []
    per_chip_accept_maps: List[Dict[float, float]] = []
    threshold_set = set()

    for chip_id in chip_ids:
        # Determine heldout count for this chip
        if chip_id not in all_files:
            # Skip unknown chips
            continue
        readouts: ReadoutList = read_readouts(all_files[chip_id])
        total_reads = len(readouts)
        # Determine number of cells for acceptance fraction
        num_cells = int(readouts[0].data.size) if total_reads > 0 else 1
        heldout_reads = max(total_reads - int(num_enroll_readings), 0)

        # Load all BER cache files for this chip and num_enroll_readings
        prefix = f"regenerate_ber_{chip_id}_"
        suffix = f"_num_readings{int(num_enroll_readings)}.pkl"
        files = _list_cache_files(cache_dir, prefix, suffix)

        chip_map: Dict[float, float] = {}
        chip_acc_map: Dict[float, float] = {}
        for fpath in files:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            # Each file contains one threshold's arrays
            threshold = float(data["threshold"])  # key
            error_count = np.asarray(data["error_count"])  # (ranges, Tdim) or (ranges, 1)
            valid_count = np.asarray(data["valid_patterns_count"])  # (ranges, Tdim) or (ranges, 1)

            # Use the processor's helper to compute BER and acceptance per entry
            # Returns ber_rate (1, ...) and acceptance_rate (1, ...)
            ber_rates, acc_rates = GlobalBERProcessor.get_rates_given_counts_single_threshold(
                (error_count, valid_count), heldout_reads, return_both=True, num_cells=num_cells
            )

            # Use the last iteration (last column) for both iterative and non-iterative cases
            # For (ranges, 1): [:, -1] gives (ranges,) - same as [:, 0]
            # For (ranges, K): [:, -1] gives (ranges,) - last iteration after K readings
            ber_last_iter = ber_rates[0, :, -1] if ber_rates.ndim > 2 else ber_rates[0, :]
            acc_last_iter = acc_rates[0, :, -1] if acc_rates.ndim > 2 else acc_rates[0, :]
            
            # Average over ranges only (not over iterations)
            ber_scalar = float(np.mean(ber_last_iter)) if ber_last_iter.size else 0.0
            chip_map[threshold] = ber_scalar
            threshold_set.add(threshold)

            acc_scalar = float(np.mean(acc_last_iter)) if acc_last_iter.size else 0.0
            chip_acc_map[threshold] = acc_scalar

        if chip_map:
            per_chip_maps.append(chip_map)
            per_chip_accept_maps.append(chip_acc_map)

    if not per_chip_maps:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    thresholds = np.array(sorted(threshold_set), dtype=float)

    # Build matrix [num_chips, num_thresholds] with NaNs for missing entries
    num_chips = len(per_chip_maps)
    mat = np.full((num_chips, thresholds.size), np.nan, dtype=float)
    for i, cmap in enumerate(per_chip_maps):
        for j, th in enumerate(thresholds):
            if th in cmap:
                mat[i, j] = cmap[th]

    mean_ber = np.nanmean(mat, axis=0)
    std_ber = np.nanstd(mat, axis=0)

    # Build acceptance matrices
    acc_mat = np.full((len(per_chip_accept_maps), thresholds.size), np.nan, dtype=float)
    for i, cmap in enumerate(per_chip_accept_maps):
        for j, th in enumerate(thresholds):
            if th in cmap:
                acc_mat[i, j] = cmap[th]
    mean_accept = np.nanmean(acc_mat, axis=0)
    std_accept = np.nanstd(acc_mat, axis=0)
    return thresholds, mean_ber, std_ber, mean_accept, std_accept


def aggregate_helper_equal_ranges_over_chips(
    chip_ids: List[str],
    K: int,
    cache_dir: Path = Path("previous_work/helperless_stabilizer_bernardini/experiments/cache"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate HelperDataComparator equal-ranges results over chips (D = K mode).

    For each chip, reads grouped cache files enroll_comparator_{chip}_K{K}_delta{T}_equal.pkl
    (multiple thresholds per file). For each threshold, it computes per-range per-iteration
    BER as error_count / (accepted_cells * K), averages over ranges and iterations, then
    averages over chips.

    Returns:
        thresholds: np.ndarray of thresholds (sorted)
        mean_ber: np.ndarray of mean BER per threshold
        std_ber: np.ndarray of std BER per threshold
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    per_chip_maps: List[Dict[float, float]] = []
    threshold_set = set()

    # Scan cache directory for this K
    for chip_id in chip_ids:
        chip_map: Dict[float, float] = {}
        # Files for this chip & mode equal
        prefix = f"enroll_comparator_{chip_id}_K{int(K)}_delta"
        suffix = "_equal.pkl"
        for fpath in _list_cache_files(cache_dir, prefix, suffix):
            with open(fpath, "rb") as f:
                grouped = pickle.load(f)  # dict keyed by threshold D
            for th_str, rec in grouped.items():
                # Keys may be float or str representing threshold
                try:
                    th = float(th_str)
                except (TypeError, ValueError):
                    continue
                err = np.asarray(rec.get("error_count", []))  # (ranges, K)
                acc = np.asarray(rec.get("accepted_cells_count", []))  # (ranges,)
                if err.size == 0 or acc.size == 0:
                    continue
                # Compute per-entry BER using the processor helper:
                # treat K as the "test_readings_count" and broadcast accepted per range
                K_int = int(K)
                if err.ndim == 2:
                    valid_broadcast = np.repeat(acc[:, None], err.shape[1], axis=1)
                else:
                    valid_broadcast = acc
                ber = GlobalBERProcessor.get_rates_given_counts_single_threshold(
                    (err, valid_broadcast), K_int
                )[0]
                # Average over ranges and iterations
                ber_scalar = float(np.mean(ber)) if ber.size else 0.0
                chip_map[th] = ber_scalar
                threshold_set.add(th)

        if chip_map:
            per_chip_maps.append(chip_map)

    if not per_chip_maps:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.array(sorted(threshold_set), dtype=float)
    num_chips = len(per_chip_maps)
    mat = np.full((num_chips, thresholds.size), np.nan, dtype=float)
    for i, cmap in enumerate(per_chip_maps):
        for j, th in enumerate(thresholds):
            if th in cmap:
                mat[i, j] = cmap[th]

    mean_ber = np.nanmean(mat, axis=0)
    std_ber = np.nanstd(mat, axis=0)
    return thresholds, mean_ber, std_ber


def aggregate_helper_keqne_series_over_chips(
    chip_ids: List[str],
    K: int,
    cache_dir: Path = Path("previous_work/helperless_stabilizer_bernardini/experiments/cache"),
) -> Dict[Tuple[int, float], Dict[str, np.ndarray]]:
    """Aggregate HelperDataComparator non-equal mode (fixed (K, delta) vs (N, D) series).

    For each chip, reads grouped cache files enroll_comparator_{chip}_K{K}_delta{delta}_keqne_series.pkl
    keyed by (N, D). Each entry contains enrollment_ber_error_counts (len N) and
    enrollment_ber_compared_cells (len N). We compute per-iteration BER series as
    errors / compared, aggregate over chips for each (N, D).

    Returns a dict keyed by (N, D) -> { 't': np.arange(1..N), 'mean_ber': array(N), 'std_ber': array(N) }.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect per (N, D) lists of series from chips
    series_map: Dict[Tuple[int, float], List[np.ndarray]] = {}

    for chip_id in chip_ids:
        prefix = f"enroll_comparator_{chip_id}_K{int(K)}_delta"
        suffix = "_keqne_series.pkl"
        for fpath in _list_cache_files(cache_dir, prefix, suffix):
            with open(fpath, "rb") as f:
                grouped = pickle.load(f)  # dict keyed by (N, D)
            for key, rec in grouped.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                N_val, D_val = int(key[0]), float(key[1])
                err = np.asarray(rec.get("enrollment_ber_error_counts", []))  # (N,)
                cmp_cells = np.asarray(rec.get("enrollment_ber_compared_cells", []))  # (N,)
                if err.size == 0 or cmp_cells.size == 0:
                    continue
                # Use comparator helper to compute BER series
                ber_series = HelperDataComparator.series_rates_from_counts(err, cmp_cells)
                series_map.setdefault((N_val, D_val), []).append(ber_series)

    # Aggregate across chips per (N, D)
    out: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    for (N_val, D_val), lst in series_map.items():
        # Pad/crop to length N_val for safety
        arr = np.stack([x[:N_val] for x in lst], axis=0)  # (num_chips, N)
        mean_ber = np.nanmean(arr, axis=0)
        std_ber = np.nanstd(arr, axis=0)
        t = np.arange(1, N_val + 1, dtype=int)
        out[(N_val, D_val)] = {"t": t, "mean_ber": mean_ber, "std_ber": std_ber}

    return out



def load_incremental_enrollment_ber_per_chip(
    K: int,
    delta: float,
    D: float | None = None,
    cache_dir: Path = Path("previous_work/helperless_stabilizer_bernardini/experiments/cache"),
) -> Dict[str, Dict[str, List[float]]]:
    """Load incremental enrollment BER series per chip (averaged over ranges).

    Reads cache files written by ComparatorCacheManager.save_incremental_enrollment_ber:
      incremental_enroll_ber_{chip_id}_K{K}_delta{delta}_D{D}.pkl

    For each chip:
      - Averages error_count over ranges to get a length-K error series
      - Divides by num_cells for that chip to get a BER series
      - Computes Bit Selection Rate (BSR) as accepted_cells_fraction per iteration:
        bsr(t) = 1 - discarded_cells_count(t) / num_cells, averaged over ranges
      - Returns {'iterations': [1..K], 'ber_mean': [...], 'bsr_mean': [...]}

    Args:
        K: Enrollment/test readings per range
        delta: Reference threshold used for the base mask
        D: Test threshold. If None, uses delta.
        cache_dir: Directory where cache files are stored

    Returns:
        dict: { chip_id: {
                'iterations': List[int],
                'ber_mean': List[float],
                'bsr_mean': List[float],   # Bit Selection Rate = 1 - discarded_fraction
            } }
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    d_str = f"{float(delta):.3f}".replace('.', 'p')
    use_D = delta if D is None else D
    D_str = f"{float(use_D):.3f}".replace('.', 'p')

    # filename template: incremental_enroll_ber_{chip_id}_K{K}_delta{d_str}_D{D_str}.pkl
    # collect all matching and prefer exact D when multiple
    results: Dict[str, Dict[str, List[float]]] = {}

    # We need num_cells per chip
    all_files = get_files()

    for p in cache_dir.iterdir():
        name = p.name
        if not name.startswith("incremental_enroll_ber_"):
            continue
        if f"_K{int(K)}_" not in name:
            continue
        if f"_delta{d_str}_" not in name:
            continue
        if not name.endswith(".pkl"):
            continue

        parts = name.split("_")
        # incremental_enroll_ber, {chip}, K{K}, delta{d_str}, D{...}.pkl
        if len(parts) < 5:
            continue
        chip_id = parts[3]

        # prefer exact D match
        if not name.endswith(f"_D{D_str}.pkl"):
            # if a better match might come later, skip non-matching Ds for now
            # but only if no entry yet; otherwise keep the first found
            if chip_id in results:
                continue

        # Load cache
        with open(p, "rb") as f:
            obj = pickle.load(f)
        err_counts = np.asarray(obj.get("error_count", []), dtype=np.float64)  # (ranges, K)
        # Cached field name uses 'patterns', but these counts represent discarded cells
        discarded_cells_counts = np.asarray(obj.get("discarded_patterns_count", []), dtype=np.float64)  # (ranges, K)
        if err_counts.ndim != 2 or err_counts.shape[1] != int(K):
            continue
        if discarded_cells_counts.ndim != 2 or discarded_cells_counts.shape[1] != int(K):
            # If discarded not present or malformed, skip BSR computation for this chip
            discarded_cells_counts = None

        # Determine num_cells for this chip
        if chip_id not in all_files:
            continue
        readouts = read_readouts(all_files[chip_id])
        # number of cells (bits) per read
        num_cells = int(readouts[0].data.size) if len(readouts) > 0 else 1
        if num_cells <= 0:
            num_cells = 1

        # Average over ranges only
        err_avg = np.mean(err_counts, axis=0)  # (K,)
        ber_series = (err_avg / float(num_cells)).tolist()

        # Bit Selection Rate: accepted cells fraction per iteration, averaged over ranges
        if discarded_cells_counts is not None:
            disc_avg = np.mean(discarded_cells_counts, axis=0)  # (K,)
            bsr_series = (1.0 - (disc_avg / float(num_cells))).tolist()
        else:
            bsr_series = []
        iters = list(range(1, int(K) + 1))

        results[chip_id] = {"iterations": iters, "ber_mean": ber_series, "bsr_mean": bsr_series}

    return results
