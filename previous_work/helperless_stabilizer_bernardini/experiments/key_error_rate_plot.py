"""Key error rate vs key length plot for Bernardini helperless stabilizer.

Sweeps over multiple key lengths for one or more (K_ref, delta_ref, delta_test)
configurations, computes empirical KER plus analytical estimators, and plots
failure rate vs key length.  Mirrors the structure of the nvm_free_tmvs
key_error_rate_plot.py but works with the Bernardini enrollment model.

Usage (from repository root, venv active):
    python -m previous_work.helperless_stabilizer_bernardini.experiments.key_error_rate_plot
"""

import os
import time
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from previous_work.helperless_stabilizer_bernardini.experiments.key_error_rate_processor import (
    BernardiniKeyErrorRateProcessor,
    key_failure_formula1,
    key_failure_formula2,
    key_failure_closed_form,
    key_failure_formula3_enrollment,
    key_failure_formula4_pselect,
)
from common.data_reading_utils import get_files, read_readouts, ReadoutList


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Key lengths to sweep (x-axis)
KEY_LENGTHS: List[int] = [1, 4, 8, 16, 32, 64, 128]

# Each tuple: (K_ref, delta_ref, delta_test, label)
PARAMETERS: List[Tuple[int, float, float, str]] = [
    (500, 0.499, 0.499, "K=500,d=0.499"),
]

# Limit number of chips to keep experiment tractable.
MAX_CHIPS: int = 9

# If True, ignore cached CSV results and recompute everything.
FORCE_RECOMPUTE: bool = False

# Directory for saving per-chip results from the sweep.
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "methods_experiments_data"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key(
    chip_id: str,
    K_ref: int,
    delta_ref: float,
    delta_test: float,
    key_length: int,
):
    """Build a hashable key for caching based on configuration and chip."""
    return (
        str(chip_id),
        str(K_ref),
        str(delta_ref),
        str(delta_test),
        str(key_length),
    )


def _load_existing_results() -> Dict[tuple, Dict]:
    """Load previously computed per-chip results from CSV files, if any."""
    existing: Dict[tuple, Dict] = {}
    if not os.path.isdir(RESULTS_DIR):
        return existing

    for fname in os.listdir(RESULTS_DIR):
        if not (fname.startswith("key_error_results_") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = _make_key(
                        row["chip_id"],
                        row["K_ref"],
                        row["delta_ref"],
                        row["delta_test"],
                        row["key_length"],
                    )

                    count_errors_all = float(row["count_errors_all"])
                    count_errors_sel_changed = float(row["count_errors_sel_changed"])
                    count_selected_ref = float(row["count_selected_ref"])
                    count_patterns = float(row["count_patterns"])
                    count_test_ranges = float(row["count_test_ranges"])
                    count_key_errors = float(row["count_key_errors"])
                    count_total_keys = float(row["count_total_keys"])
                    key_length = float(row["key_length"])

                    # Recompute rates from counts
                    if count_test_ranges > 0 and count_patterns > 0:
                        ber = count_errors_all / (count_patterns * count_test_ranges)
                    else:
                        ber = 0.0

                    if count_test_ranges > 0 and count_selected_ref > 0:
                        ber_sel_changed = count_errors_sel_changed / (
                            count_selected_ref * count_test_ranges
                        )
                        ber_all_over_selected = count_errors_all / (
                            count_selected_ref * count_test_ranges
                        )
                    else:
                        ber_sel_changed = 0.0
                        ber_all_over_selected = 0.0

                    if count_total_keys > 0:
                        key_error_rate = count_key_errors / count_total_keys
                    else:
                        key_error_rate = 0.0

                    if count_patterns > 0:
                        p_select = count_selected_ref / count_patterns
                    else:
                        p_select = 0.0

                    # Split BER
                    p_selected = p_select
                    if count_selected_ref > 0 and count_test_ranges > 0:
                        b_s = count_errors_sel_changed / (
                            count_selected_ref * count_test_ranges
                        )
                    else:
                        b_s = 0.0

                    count_unselected = count_patterns - count_selected_ref
                    total_errors_unsel = count_errors_all - count_errors_sel_changed
                    if count_unselected > 0 and count_test_ranges > 0:
                        b_u = total_errors_unsel / (count_unselected * count_test_ranges)
                    else:
                        b_u = 0.0

                    ber_split = p_selected * b_s + (1.0 - p_selected) * b_u

                    existing[key] = {
                        "key_error_rate": key_error_rate,
                        "ber": ber,
                        "ber_sel_changed": ber_sel_changed,
                        "ber_all_over_selected": ber_all_over_selected,
                        "ber_split": ber_split,
                        "p_select": p_select,
                        "b_s": b_s,
                        "b_u": b_u,
                        "formula1": key_failure_formula1(ber, key_length),
                        "formula2": key_failure_formula2(ber, p_select, key_length),
                        "formula1_sel_changed": key_failure_formula1(ber_sel_changed, key_length),
                        "formula1_split": key_failure_formula1(ber_split, key_length),
                        "formula_closed_form": key_failure_closed_form(ber, p_select, key_length),
                        "formula3": key_failure_formula3_enrollment(b_s, b_u, p_select, key_length),
                        "formula4": key_failure_formula4_pselect(p_select, key_length),
                    }
                except (KeyError, ValueError):
                    continue

    return existing


def aggregate_means(per_chip_results: List[Dict], key_length: int) -> Dict[str, float]:
    """Aggregate per-chip results using averaged parameters, not averaged formulas."""
    n_chips = len(per_chip_results)
    if n_chips == 0:
        return {}

    def avg(k: str) -> float:
        return sum(r[k] for r in per_chip_results) / n_chips

    ker_mean = avg("key_error_rate")
    ber_mean = avg("ber")
    ber_sel_changed_mean = avg("ber_sel_changed")
    ber_all_over_selected_mean = avg("ber_all_over_selected")
    ber_split_mean = avg("ber_split")
    p_select_mean = avg("p_select")
    b_s_mean = avg("b_s")
    b_u_mean = avg("b_u")

    return {
        "key_error_rate": ker_mean,
        "formula1": key_failure_formula1(ber_mean, key_length),
        "formula2": key_failure_formula2(ber_mean, p_select_mean, key_length),
        "formula1_sel_changed": key_failure_formula1(ber_sel_changed_mean, key_length),
        "formula1_split": key_failure_formula1(ber_split_mean, key_length),
        "formula_closed_form": key_failure_closed_form(ber_mean, p_select_mean, key_length),
        "formula3": key_failure_formula3_enrollment(b_s_mean, b_u_mean, p_select_mean, key_length),
        "formula4": key_failure_formula4_pselect(p_select_mean, key_length),
    }


def run_sweep() -> Dict[Tuple[str, int], Dict[str, float]]:
    """Run the KER experiment for multiple key lengths and configurations."""
    existing_results = {} if FORCE_RECOMPUTE else _load_existing_results()

    all_files = get_files()
    chip_ids = list(all_files.keys())[:MAX_CHIPS]
    all_readouts: Dict[str, ReadoutList] = {
        cid: read_readouts(all_files[cid]) for cid in chip_ids
    }

    results_means: Dict[Tuple[str, int], Dict[str, float]] = {}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"key_error_results_{timestamp}.csv")

    fieldnames = [
        "chip_id",
        "K_ref",
        "delta_ref",
        "delta_test",
        "label",
        "key_length",
        "count_errors_all",
        "count_errors_sel_changed",
        "count_selected_ref",
        "count_patterns",
        "count_test_ranges",
        "count_key_errors",
        "count_total_keys",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for K_ref, delta_ref, delta_test, cfg_label in PARAMETERS:
            print(f"\nConfig {cfg_label}: K_ref={K_ref}, delta_ref={delta_ref}, delta_test={delta_test}")

            for key_length_test in KEY_LENGTHS:
                print(f"  Key length K={key_length_test}")
                per_chip_results: List[Dict] = []

                for cid, readouts_val in all_readouts.items():
                    key = _make_key(cid, K_ref, delta_ref, delta_test, key_length_test)

                    if not FORCE_RECOMPUTE and key in existing_results:
                        per_chip_results.append(existing_results[key])
                        print(f"    Chip {cid} ... cached")
                        continue

                    print(f"    Chip {cid} ...", end="", flush=True)
                    t0 = time.time()

                    processor = BernardiniKeyErrorRateProcessor(
                        readouts_val,
                        num_enroll_readings=K_ref,
                        delta_ref=delta_ref,
                        delta_test=delta_test,
                    )
                    res = processor.run(key_length_test=key_length_test)
                    per_chip_results.append(res)

                    writer.writerow({
                        "chip_id": cid,
                        "K_ref": K_ref,
                        "delta_ref": delta_ref,
                        "delta_test": delta_test,
                        "label": cfg_label,
                        "key_length": key_length_test,
                        "count_errors_all": res["count_errors_all"],
                        "count_errors_sel_changed": res["count_errors_sel_changed"],
                        "count_selected_ref": res["count_selected_ref"],
                        "count_patterns": res["count_patterns"],
                        "count_test_ranges": res["count_test_ranges"],
                        "count_key_errors": res["count_key_errors"],
                        "count_total_keys": res["count_total_keys"],
                    })

                    dt = time.time() - t0
                    print(f" done ({dt:.1f}s)")

                means = aggregate_means(per_chip_results, key_length_test)
                results_means[(cfg_label, key_length_test)] = means

    return results_means


def plot_results(results_means: Dict[Tuple[str, int], Dict[str, float]]) -> None:
    """Plot different estimators vs key length for each configuration."""
    estimators = [
        ("Empirical", "key_error_rate", "k"),
        ("F1_raw", "formula1", "C0"),
        ("F2_raw", "formula2", "C1"),
        ("F1_selChg", "formula1_sel_changed", "C2"),
        ("F1_split", "formula1_split", "C4"),
        ("F_cf", "formula_closed_form", "C3"),
        ("F3_enr", "formula3", "C5"),
        ("F4_psel", "formula4", "C6"),
    ]

    markers = ["o", "s", "^", "D", "P", "X"]

    plt.figure(figsize=(8, 5))

    for cfg_idx, param in enumerate(PARAMETERS):
        cfg_label = param[-1]
        marker = markers[cfg_idx % len(markers)]

        for est_name, key_in_results, color in estimators:
            y_vals: List[float] = []
            for K in KEY_LENGTHS:
                means = results_means.get((cfg_label, K))
                if not means:
                    y_vals.append(float("nan"))
                else:
                    y_vals.append(means[key_in_results])

            plt.plot(
                KEY_LENGTHS,
                y_vals,
                linestyle="-",
                marker=marker,
                color=color,
                label=f"{cfg_label}-{est_name}",
            )

    plt.xlabel("Key length K")
    plt.ylabel("Key error rate / estimates")
    plt.xticks(KEY_LENGTHS, [str(k) for k in KEY_LENGTHS])
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_sweep()
    plot_results(results)
