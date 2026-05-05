"""Key error rate experiment for helperless stabilizer Bernardini approach.

This script mirrors the NVM-free TMVS key error rate experiment, but works
directly on bit matrices and Bernardini-style acceptance masks.

High-level idea per chip:
  1. Build bit matrix of shape (num_reads, num_cells).
  2. For a fixed (K_ref, delta_ref), pick the first enrollment range as the
     reference. Compute:
       - reference acceptance mask (which cells contribute to the key)
       - reference majority bits for those cells
  3. From the reference mask, traverse cells in order and extract consecutive
     full keys of length key_length_test, skipping rejected cells. This yields
     a list of [start_cell_idx, end_cell_idx] boundaries in cell index space.
  4. For each other enrollment range (same K_ref and delta_ref), compute its
     mask/majority and build a combined enrollment state (-1/0/1) that
     mirrors the nvm_free_tmvs format (-1 rejected, 0/1 accepted bit).
     Compare against the reference state:
       - A full key is counted as erroneous if any cell within its boundary
         has a different combined state (mask change or bit-value change).
       - Per-cell BERs are also computed (over all cells and over selected
         cells only), as in the NVM-free TMVS experiment.
  5. From these, compute empirical KER plus several analytical estimators
     (F1/F2/F1_adj/F2_adj/F1_selChg/F1_split/F_cf/F3_enr).
"""

from __future__ import annotations

import time
from typing import List, Tuple, Dict

import numpy as np

from common.data_reading_utils import get_files, read_readouts, ReadoutList
from previous_work.helperless_stabilizer_bernardini.evaluate_ber import (
    build_bit_matrix,
    mask_from_threshold,
    majority_from_matrix,
)
from previous_work.helperless_stabilizer_bernardini.experiments.analysis_utils import (
    get_enrollment_ranges,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_enrollment_state(
    mask: np.ndarray,
    majority: np.ndarray,
) -> np.ndarray:
    """Build combined enrollment state mirroring the nvm_free_tmvs format.

    Returns an int8 array where:
      -1 = rejected (mask is False)
       0 = accepted, majority vote is 0
       1 = accepted, majority vote is 1

    This ensures that comparing ``ref_state != test_state`` captures both
    mask changes (accepted <-> rejected) and bit-value changes (0 <-> 1),
    exactly like ``ref_slice != test_slice`` in the nvm_free_tmvs experiment.
    """
    state = np.full(mask.shape, -1, dtype=np.int8)
    state[mask] = majority[mask].astype(np.int8)
    return state


def extract_key_boundaries(
    ref_mask: np.ndarray,
    key_length_test: int,
) -> List[Tuple[int, int]]:
    """Extract consecutive full-key boundaries from the reference mask.

    We walk the cell indices in order. Every time we see an accepted cell in
    the reference mask we advance the bit counter. When we accumulate
    key_length_test accepted cells, we close a key at the current cell index.
    The boundary is expressed as (start_cell_idx, end_cell_idx), both
    inclusive, in terms of the *cell index* (not only accepted positions).
    """
    assert ref_mask.ndim == 1
    num_cells = ref_mask.shape[0]
    boundaries: List[Tuple[int, int]] = []

    current_start: int | None = None
    bits_collected = 0

    for idx in range(num_cells):
        if ref_mask[idx]:
            if bits_collected == 0:
                current_start = idx
            bits_collected += 1

            if bits_collected == key_length_test:
                assert current_start is not None
                boundaries.append((current_start, idx))
                # Reset for next key, starting after this cell
                current_start = None
                bits_collected = 0

    return boundaries


def compute_key_errors(
    ref_state: np.ndarray,
    test_state: np.ndarray,
    key_boundaries: List[Tuple[int, int]],
) -> int:
    """Count how many full keys have at least one state mismatch.

    Compares the combined enrollment state (-1/0/1) for all cells within each
    key boundary, including rejected (-1) positions.  A disagreement on a
    rejected position means the test enrollment changed its selection, which
    is enough to corrupt the key — matching the nvm_free_tmvs semantics where
    -1 (discarded) positions are also compared.
    """
    errors = 0
    for start, end in key_boundaries:
        if np.any(ref_state[start : end + 1] != test_state[start : end + 1]):
            errors += 1
    return errors


# ---------------------------------------------------------------------------
# Failure formulas (copied/adapted from nvm_free_tmvs key_error_rate_processor)
# ---------------------------------------------------------------------------

def key_failure_formula1(ber: float, key_length: int) -> float:
    """Failure estimate 1: 1 - (1 - ber)^key_length."""
    if key_length <= 0 or ber >= 1.0:
        return 1.0
    return 1.0 - (1.0 - ber) ** key_length


def key_failure_formula2(ber: float, p_select: float, key_length: int) -> float:
    """Failure estimate 2: 1 - (1 - ber)^(key_length / p_select)."""
    if p_select <= 0 or key_length <= 0 or ber >= 1.0:
        return 1.0
    exponent = key_length / p_select
    return 1.0 - (1.0 - ber) ** exponent


def key_failure_closed_form(ber: float, p_select: float, key_length: int) -> float:
    """Closed-form KER estimate (simplified enrollment model).

    P_ok = ((p * (1 - b)) / (p + b - p * b)) ** K
    P_fail = 1 - P_ok

    where p is p_select, b is ber, and K is key_length.
    """
    if key_length <= 0 or p_select <= 0 or ber >= 1.0:
        return 1.0
    denom = p_select + ber - p_select * ber
    if denom <= 0:
        return 1.0
    ratio = (p_select * (1.0 - ber)) / denom
    ratio = max(0.0, min(1.0, ratio))
    p_ok = ratio ** key_length
    return 1.0 - p_ok


def key_failure_formula3_enrollment(
    b_s: float, b_u: float, p_select: float, key_length: int
) -> float:
    """Enrollment failure probability (recommended model):

    P_fail,enr = 1 - (1 - b_s)^K (1 - b_u)^(K(1/p - 1))
    """
    if key_length <= 0 or p_select <= 0 or b_s >= 1.0 or b_u >= 1.0:
        return 1.0
    exponent_unselected = key_length * (1.0 / p_select - 1.0)
    p_ok = (1.0 - b_s) ** key_length * (1.0 - b_u) ** exponent_unselected
    return 1.0 - p_ok


def key_failure_formula4_pselect(p_select: float, key_length: int) -> float:
    """Failure estimate 4: 1 - p_select^(key_length / p_select).

    Models the probability that at least one of the K / p_select patterns
    examined to collect K key bits was not selected (i.e. changed selection
    state between enrollments).
    """
    if key_length <= 0 or p_select <= 0 or p_select >= 1.0:
        return 1.0
    exponent = key_length / p_select
    return 1.0 - p_select ** exponent


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------

class BernardiniKeyErrorRateProcessor:
    """Key error rate experiment for a single chip under Bernardini approach.

    Parameters
    ----------
    readouts:
        Chip readouts.
    num_enroll_readings:
        Number of enrollment readings per range (K_ref).
    delta_ref:
        Threshold used for the *reference* enrollment (δ in the paper).
    delta_test:
        Threshold used for the *test* enrollments (D in the paper). If None,
        the same value as delta_ref is used (symmetric case).
    """

    def __init__(
        self,
        readouts: ReadoutList,
        num_enroll_readings: int,
        delta_ref: float,
        delta_test: float | None = None,
    ):
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.num_enroll_readings = int(num_enroll_readings)
        self.delta_ref = float(delta_ref)
        self.delta_test = float(delta_ref if delta_test is None else delta_test)

    def run(self, key_length_test: int) -> Dict[str, float]:
        """Run the experiment for this chip and return aggregated metrics.

        Returns a dict with empirical KER, BER variants, selection rate, number
        of full keys, and the different failure estimates.
        """
        bit_matrix = build_bit_matrix(self.readouts)
        num_reads, num_cells = bit_matrix.shape

        enroll_ranges = get_enrollment_ranges(self.num_enroll_readings)
        if enroll_ranges.shape[0] < 2:
            raise ValueError(
                "Not enough enrollment ranges for reference + test in this configuration."
            )

        ref_start, ref_end = map(int, enroll_ranges[0])
        test_ranges = [tuple(map(int, r)) for r in enroll_ranges[1:]]

        # Reference enrollment
        ref_sub = bit_matrix[ref_start:ref_end, :]
        K_ref = ref_sub.shape[0]
        if K_ref != self.num_enroll_readings:
            raise ValueError(
                f"Reference range length {K_ref} != num_enroll_readings {self.num_enroll_readings}"
            )

        ref_mask = mask_from_threshold(ref_sub, K_ref, self.delta_ref)
        ref_majority = majority_from_matrix(ref_sub, K_ref)
        ref_state = build_enrollment_state(ref_mask, ref_majority)

        p_select_ref = float(np.mean(ref_mask))
        key_boundaries = extract_key_boundaries(ref_mask, key_length_test)
        num_full_keys = len(key_boundaries)

        # If no full keys, return zeros to avoid division-by-zero noise
        if num_full_keys == 0:
            return {
                "key_error_rate": 0.0,
                "ber": 0.0,
                "ber_sel_changed": 0.0,
                "ber_all_over_selected": 0.0,
                "ber_split": 0.0,
                "p_select": p_select_ref,
                "num_full_keys": 0,
                "formula1": 0.0,
                "formula2": 0.0,
                "formula1_adj": 0.0,
                "formula2_adj": 0.0,
                "formula1_sel_changed": 0.0,
                "formula1_all_over_selected": 0.0,
                "formula1_split": 0.0,
                "formula_closed_form": 0.0,
                "formula3": 0.0,
                "formula4": 0.0,
            }

        # Accumulators over all test ranges
        range_key_errors = 0
        total_errors_all = 0
        total_errors_sel_changed = 0

        pattern_selected_ref = ref_mask.astype(bool)
        num_selected_ref = int(np.sum(pattern_selected_ref))
        num_patterns = num_cells

        for start_idx, end_idx in test_ranges:
            test_sub = bit_matrix[start_idx:end_idx, :]
            if test_sub.shape[0] != K_ref:
                # Skip malformed ranges
                continue

            test_mask = mask_from_threshold(test_sub, K_ref, self.delta_test)
            test_majority = majority_from_matrix(test_sub, K_ref)
            test_state = build_enrollment_state(test_mask, test_majority)

            # Key errors (compare combined states, matching nvm_free_tmvs)
            range_key_errors += compute_key_errors(ref_state, test_state, key_boundaries)

            # Per-cell change mask (combined state captures both mask
            # changes and bit-value changes, equivalent to
            # np.any(ref_slice != test_slice, axis=-1) in nvm_free_tmvs)
            pattern_changed = ref_state != test_state

            total_errors_all += int(np.sum(pattern_changed))
            total_errors_sel_changed += int(np.sum(pattern_changed & pattern_selected_ref))

        num_test_ranges = len(test_ranges)
        range_total_keys = num_full_keys * num_test_ranges

        # BER definitions
        if num_test_ranges and num_patterns:
            avg_ber = total_errors_all / (num_patterns * num_test_ranges)
        else:
            avg_ber = 0.0

        if num_test_ranges and num_selected_ref:
            ber_sel_changed = total_errors_sel_changed / (
                num_selected_ref * num_test_ranges
            )
            ber_all_over_selected = total_errors_all / (
                num_selected_ref * num_test_ranges
            )
        else:
            ber_sel_changed = 0.0
            ber_all_over_selected = 0.0

        # Split BER components
        if num_patterns > 0:
            p_selected = num_selected_ref / num_patterns
        else:
            p_selected = 0.0

        if num_selected_ref > 0 and num_test_ranges > 0:
            b_s = total_errors_sel_changed / (num_selected_ref * num_test_ranges)
        else:
            b_s = 0.0

        num_unselected = num_patterns - num_selected_ref
        total_errors_unsel_changed = total_errors_all - total_errors_sel_changed
        if num_unselected > 0 and num_test_ranges > 0:
            b_u = total_errors_unsel_changed / (num_unselected * num_test_ranges)
        else:
            b_u = 0.0

        ber_split = p_selected * b_s + (1.0 - p_selected) * b_u

        # Empirical KER
        if range_total_keys > 0:
            avg_key_error_rate = range_key_errors / range_total_keys
        else:
            avg_key_error_rate = 0.0

        # Analytical estimates (use same shapes as NVM-free TMVS KER script)
        ber_raw = avg_ber
        # No codebook dimension here, so "adjusted" BER is identical
        ber_adj = ber_raw

        f1 = key_failure_formula1(ber_raw, key_length_test)
        f2 = key_failure_formula2(ber_raw, p_select_ref, key_length_test)
        f1_adj = key_failure_formula1(ber_adj, key_length_test)
        f2_adj = key_failure_formula2(ber_adj, p_select_ref, key_length_test)
        f1_sel_changed = key_failure_formula1(ber_sel_changed, key_length_test)
        f1_all_over_selected = key_failure_formula1(
            ber_all_over_selected, key_length_test
        )
        f1_split = key_failure_formula1(ber_split, key_length_test)
        f_cf = key_failure_closed_form(ber_raw, p_select_ref, key_length_test)
        f3 = key_failure_formula3_enrollment(b_s, b_u, p_select_ref, key_length_test)
        f4 = key_failure_formula4_pselect(p_select_ref, key_length_test)

        return {
            "key_error_rate": avg_key_error_rate,
            # Raw counts for maximum flexibility when post-processing
            "count_errors_all": total_errors_all,
            "count_errors_sel_changed": total_errors_sel_changed,
            "count_selected_ref": num_selected_ref,
            "count_patterns": num_patterns,
            "count_test_ranges": num_test_ranges,
            "count_key_errors": range_key_errors,
            "count_total_keys": range_total_keys,
            "ber": avg_ber,
            "ber_sel_changed": ber_sel_changed,
            "ber_all_over_selected": ber_all_over_selected,
            "ber_split": ber_split,
            "b_s": b_s,
            "b_u": b_u,
            "p_select": p_select_ref,
            "num_full_keys": num_full_keys,
            "formula1": f1,
            "formula2": f2,
            "formula1_adj": f1_adj,
            "formula2_adj": f2_adj,
            "formula1_sel_changed": f1_sel_changed,
            "formula1_all_over_selected": f1_all_over_selected,
            "formula1_split": f1_split,
            "formula_closed_form": f_cf,
            "formula3": f3,
            "formula4": f4,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse as _ap
    from pathlib import Path as _Path
    from nvm_free_tmvs.utils.experiment_cache import get_cache_path, save_cache, load_cache

    _CACHE_BASE = _Path(__file__).resolve().parent.parent / "methods_experiments_data"

    _parser = _ap.ArgumentParser(description="Bernardini Dark Bit KER processor")
    _parser.add_argument("--force-recompute", action="store_true",
                         help="Ignore cached results and recompute.")
    _cli = _parser.parse_args()

    # Example configuration; adjust as needed.
    K_ref = 500            # Reference enrollment reads
    delta_ref = 0.499      # Reference threshold δ
    delta_test = 0.4991    # Test threshold D (can differ from δ)
    key_length_test = 128

    all_files = get_files()
    chip_ids = list(all_files.keys())
    all_readouts: List[ReadoutList] = [
        read_readouts(all_files[cid]) for cid in chip_ids
    ]

    print(
        f"Bernardini KER experiment: K_ref={K_ref}, delta_ref={delta_ref}, "
        f"delta_test={delta_test}, key_length_test={key_length_test}"
    )

    all_results = []
    for readouts_val in all_readouts:
        chip_id = readouts_val.chip_id
        print(f"\n--- Chip {chip_id} ---")

        cache_config = {
            "method": "darkbit",
            "K_ref": K_ref,
            "delta_ref": delta_ref,
            "delta_test": delta_test,
            "key_length": key_length_test,
            "chip_id": chip_id,
        }
        cache_path = get_cache_path("bernardini_ker", cache_config, base=_CACHE_BASE)

        if not _cli.force_recompute and cache_path.exists():
            print(f"  Loading cached results from {cache_path.name}")
            res = load_cache(cache_path)["result"]
        else:
            t0 = time.time()
            processor = BernardiniKeyErrorRateProcessor(
                readouts_val,
                num_enroll_readings=K_ref,
                delta_ref=delta_ref,
                delta_test=delta_test,
            )
            res = processor.run(key_length_test=key_length_test)
            save_cache(cache_path, {"config": cache_config, "result": res})
            print(f"  Elapsed: {time.time() - t0:.1f}s")

        all_results.append(res)

        print(
            f"  KER={res['key_error_rate']:.6e}  "
            f"BER={res['ber']:.6e}  p_sel={res['p_select']:.4f}  "
            f"#keys={res['num_full_keys']}  "
            f"F1={res['formula1']:.6e}  F2={res['formula2']:.6e}  "
            f"F1_adj={res['formula1_adj']:.6e}  F2_adj={res['formula2_adj']:.6e}  "
            f"F1_selChg={res['formula1_sel_changed']:.6e}  "
            f"F1_all/sel={res['formula1_all_over_selected']:.6e}  "
            f"F1_split={res['formula1_split']:.6e}  "
            f"F_cf={res['formula_closed_form']:.6e}  "
            f"F3_enr={res['formula3']:.6e}  "
            f"F4={res['formula4']:.6e}"
        )

    # Pooled KER across all chips
    if len(all_results) > 1:
        total_key_errors = sum(r["count_key_errors"] for r in all_results)
        total_valid_keys = sum(r["count_total_keys"] for r in all_results)
        pooled_ker = (
            total_key_errors / total_valid_keys if total_valid_keys > 0 else 0.0
        )
        avg_ber = sum(r["ber"] for r in all_results) / len(all_results)
        avg_ps = sum(r["p_select"] for r in all_results) / len(all_results)
        print(f"\n=== Pooled KER across {len(all_results)} chips ===")
        print(
            f"  KER={pooled_ker:.6e}  "
            f"(pooled: {total_key_errors}/{total_valid_keys})  "
            f"BER={avg_ber:.6e}  p_sel={avg_ps:.4f}"
        )
