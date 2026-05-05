"""Compare regeneration BER between trivial and full codebooks.

For a single chip, loops over small code lengths and a few thresholds,
runs enrollment then regeneration BER for both trivial (2 codewords) and
full codebook, and writes results to JSONL.

Usage (from repository root, venv active):
    python -m nvm_free_tmvs.experiments.regeneration_trivial_vs_full_codebook \
        --chip L45 --nr-read 10
"""
import argparse
import json
import pathlib
import time
from datetime import datetime, timezone
from typing import List, Set, Tuple

import numpy as np

from nvm_free_tmvs.algorithm.enroll import Enroll
from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.utils.file_manager import (
    ReadoutList, get_files, read_readouts, read_codebook,
)
from nvm_free_tmvs.utils.analysis_utils import (
    get_enrollment_ranges,
    get_shifted_selection_threshold,
)
import common.data_constants as data_const

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CODE_LENGTHS: List[Tuple[int, int, int]] = [
    (3, 1, 2),
    # (7, 1, 6),
    # (9, 1, 8),
    # (11, 1, 10),
    # (13, 1, 12),
    # (15, 1, 14),
]

TRIVIAL_ONLY_N: Set[int] = {3}

# Default TH_high* values to sweep (shifted units).  For each n we pick
# a few values likely to give nonzero selection.
DEFAULT_TH_VALUES: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]

target_num_readings_list = [100]

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "experiments_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_codebook(n: int, trivial: bool, cb_coeff: List[int]):
    """Return (codebook_override_or_None, codebook_length)."""
    if trivial:
        return [0, (1 << n) - 1], 2
    cb = read_codebook(n, cb_coeff[0], cb_coeff[1])
    return None, len(cb)


def _prepare_hd(n, readouts, cb_coeff, trivial_cb):
    """Compute Hamming distances once per codebook config."""
    cdp = ChunkDataProcessor(n, readouts, True)
    chunked_data = cdp.chunk_readouts()
    hp = HammingProcessor(n, readouts, cb_coeff, True)
    hd = hp.compute_hamming_distances(
        0, data_const.READINGS_TO_ANALYZE, chunked_data,
        codebook_override=trivial_cb,
    )
    return hp, hd


def _compute_boolean_hd_sum(hd, start_idx, range_width):
    """Sum boolean(d* > 0) over all readings outside [start_idx, start_idx+range_width).

    Returns 2D array of shape (num_patterns, num_codewords).
    """
    boolean_matrix = (hd > 0).astype(np.uint16)
    end_idx = start_idx + range_width
    bool_sum = (
        np.sum(boolean_matrix[:, :, :start_idx], axis=2, dtype=np.uint16)
        + np.sum(boolean_matrix[:, :, end_idx:], axis=2, dtype=np.uint16)
    )
    return bool_sum


def _compute_regen_ber(enrollment_slice, boolean_hd_sum, num_test_readings):
    """Compute regeneration BER for a single enrollment slice.

    Parameters
    ----------
    enrollment_slice : ndarray (num_patterns, num_codewords)
        Values: 0 (bit=0), 1 (bit=1), -1 (discarded).
    boolean_hd_sum : ndarray (num_patterns, num_codewords)
        Count of test readings where d* > 0 for each (pattern, codeword).
    num_test_readings : int
        Number of test readings (READINGS_TO_ANALYZE - range_width).

    Returns (ber_reg, num_valid_bits, error_count).
    """
    valid_mask = enrollment_slice != -1
    num_valid = int(valid_mask.sum())
    if num_valid == 0:
        return 0.0, 0, 0

    # Enrolled as 0 → d* was negative → error if test reading had d* > 0
    errors_zero = np.sum(boolean_hd_sum * (enrollment_slice == 0))
    # Enrolled as 1 → d* was positive → error if test reading had d* <= 0
    errors_one = np.sum(
        (num_test_readings - boolean_hd_sum) * (enrollment_slice == 1)
    )
    total_errors = int(errors_zero + errors_one)
    ber_reg = total_errors / (num_valid * num_test_readings)
    return ber_reg, num_valid, total_errors


def _get_th_values(n, cb_low, cb_high, trivial, custom_ths):
    """Get threshold values to test.

    For trivial codebooks: any TH_high* in (0, max_pos_d_star].
    For non-trivial: enforce TH_high* >= th_high_code_star (the codebook's
    shifted positive threshold) to preserve the one-codeword-per-pattern
    guarantee.
    max_pos_d_star = n - floor(n * P_SRAM) = ceil(n/2).
    """
    floor_half = int(n * data_const.P_SRAM)
    max_pos_d_star = n - floor_half  # ceil(n/2)
    ths = custom_ths if custom_ths else list(DEFAULT_TH_VALUES)

    if trivial:
        return [th for th in ths if 0 < th <= max_pos_d_star]

    # Non-trivial: lower bound from codebook threshold
    th_high_code_star = get_shifted_selection_threshold(n, [cb_low, cb_high])[1]
    return [th for th in ths if th_high_code_star <= th <= max_pos_d_star]


# ---------------------------------------------------------------------------
# JSONL save / resume
# ---------------------------------------------------------------------------

def _config_key(n, cb_low, cb_high, trivial, th_star, nr_read, range_idx):
    return (n, cb_low, cb_high, trivial, round(th_star, 4), nr_read, range_idx)


def _load_completed(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(_config_key(
                    r["n"], r["cb_low"], r["cb_high"],
                    r["trivial_codebook"], r["TH_high_star"],
                    r["nr_read"], r["range_idx"],
                ))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_result(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    chip_id: str,
    nr_read: int,
    custom_ths: List[float] = None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / f"regen_trivial_vs_full_{chip_id}.jsonl"
    completed = _load_completed(jsonl_path)
    print(f"Loaded {len(completed)} completed configs from {jsonl_path.name}")

    all_files = get_files()
    if chip_id not in all_files:
        raise ValueError(
            f"Chip '{chip_id}' not found. Available: {sorted(all_files.keys())}"
        )
    readouts = read_readouts(all_files[chip_id])

    enroll_ranges = get_enrollment_ranges()
    range_width = int(enroll_ranges[0][1] - enroll_ranges[0][0])
    num_enroll = min(nr_read, range_width)
    num_test_readings = data_const.READINGS_TO_ANALYZE - range_width

    # HD cache: (n, cb_low, cb_high, trivial) -> (hp, hd)
    hd_cache = {}

    configs_run = 0

    for n, cb_low, cb_high in CODE_LENGTHS:
        trivial_options = [True]
        if n not in TRIVIAL_ONLY_N:
            trivial_options.append(False)

        for trivial in trivial_options:
            label = f"n={n},cb=({cb_low},{cb_high}),triv={trivial}"
            th_values = _get_th_values(n, cb_low, cb_high, trivial, custom_ths)
            if not th_values:
                max_pos = n - int(n * data_const.P_SRAM)
                if trivial:
                    print(f"\n  [{label}] no valid thresholds in (0, {max_pos}], skipping")
                else:
                    th_min = get_shifted_selection_threshold(n, [cb_low, cb_high])[1]
                    user_max = max(custom_ths) if custom_ths else max(DEFAULT_TH_VALUES)
                    print(f"\n  [{label}] skipping: min TH*={th_min} > max user TH*={user_max}")
                continue

            # Check if all configs are done
            all_done = all(
                _config_key(n, cb_low, cb_high, trivial, th, nr_read, ri)
                in completed
                for th in th_values
                for ri in range(len(enroll_ranges))
            )
            if all_done:
                print(f"\n  [{label}] all done, skipping")
                continue

            # Get or build HD
            cache_key = (n, cb_low, cb_high, trivial)
            if cache_key not in hd_cache:
                coeff = [cb_low, cb_high]
                trivial_cb, cb_len = _build_codebook(n, trivial, coeff)
                print(f"\n  [{label}] computing HD ...", end="", flush=True)
                t0 = time.time()
                hp, hd = _prepare_hd(n, readouts, coeff, trivial_cb)
                print(f" done ({time.time()-t0:.1f}s, shape={hd.shape})")
                # Sanity check: HD must be shifted d* (centered around 0)
                hd_min = float(hd.min())
                assert hd_min < 0, (
                    f"Expected shifted d* distances with negative values; "
                    f"got range [{hd_min}, {float(hd.max())}]"
                )
                hd_cache[cache_key] = (hp, hd, cb_len)
            hp, hd, cb_len = hd_cache[cache_key]

            print(f"  [{label}] thresholds={th_values}, "
                  f"codebook_len={cb_len}")

            for th_star in th_values:
                enroll_th = [-float(th_star), float(th_star)]
                range_bers = []

                for ri, (start, end) in enumerate(enroll_ranges):
                    key = _config_key(
                        n, cb_low, cb_high, trivial, th_star, nr_read, ri)
                    if key in completed:
                        continue

                    start = int(start)
                    t0 = time.time()

                    # Enrollment
                    enroll_obj = Enroll(hp, start, num_enroll, True)
                    enroll_hd = hd[:, :, start:start + num_enroll]
                    enrollment_data = enroll_obj.execute(enroll_th, enroll_hd)

                    reading_idx = num_enroll - 1
                    enrollment_slice = enrollment_data[reading_idx]

                    # Boolean HD sum over test readings
                    bool_hd_sum = _compute_boolean_hd_sum(
                        hd, start, range_width)

                    # Regeneration BER
                    ber_reg, num_bits, error_count = _compute_regen_ber(
                        enrollment_slice, bool_hd_sum, num_test_readings)

                    elapsed = time.time() - t0
                    range_bers.append(ber_reg)

                    record = {
                        "chip_id": chip_id,
                        "n": n, "cb_low": cb_low, "cb_high": cb_high,
                        "trivial_codebook": trivial,
                        "codebook_length": cb_len,
                        "TH_high_star": round(float(th_star), 4),
                        "nr_read": nr_read,
                        "range_idx": ri,
                        "ber_reg": ber_reg,
                        "num_bits": num_bits,
                        "error_count": error_count,
                        "num_reg_trials": num_test_readings,
                        "elapsed_sec": round(elapsed, 3),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    _append_result(jsonl_path, record)
                    completed.add(key)
                    configs_run += 1

                if range_bers:
                    avg_ber = np.mean(range_bers)
                    print(
                        f"    TH*={th_star:>5.1f}  "
                        f"BER_reg={avg_ber:.6e}  "
                        f"(avg over {len(range_bers)} ranges)")

    print(f"\nDone. {configs_run} new configs. Results: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare regeneration BER: trivial vs full codebook.",
    )
    parser.add_argument("--chip", required=True, help="Chip ID (e.g. L45).")
    parser.add_argument(
        "--nr-read", type=int, default=10,
        help="Number of enrollment readings (default: 10).",
    )
    parser.add_argument(
        "--th-values", type=str, default=None,
        help="Comma-separated TH_high* values (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--list-chips", action="store_true",
        help="List available chip IDs and exit.",
    )

    args = parser.parse_args()

    if args.list_chips:
        print("Available chips:", sorted(get_files().keys()))
        return

    custom_ths = None
    if args.th_values:
        custom_ths = [float(x.strip()) for x in args.th_values.split(",")]

    run(chip_id=args.chip, nr_read=args.nr_read, custom_ths=custom_ths)


if __name__ == "__main__":
    main()
