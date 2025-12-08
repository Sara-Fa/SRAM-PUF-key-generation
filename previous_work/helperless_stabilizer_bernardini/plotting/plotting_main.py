"""Interactive plotting entrypoint for Bernardini results.

Provides a menu to select which plot to generate, mirroring the
interactive style used in `nvm_free_tmvs/plotting/plotting_main.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys
import questionary

# Ensure repository root is on sys.path for absolute imports when run as a script
repo_root = Path(__file__).parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import previous_work.helperless_stabilizer_bernardini.plotting.plotting_configuration as config
from common.data_reading_utils import get_files


def _ensure_output_dir() -> Path:
    base = Path(__file__).parent.parent / "results" / "plots"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _parse_int_list(csv_text: str) -> List[int]:
    parts = [p.strip() for p in str(csv_text).split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out


def main():
    all_files = get_files()
    chips = list(all_files.keys())
    if not chips:
        print("No chips found to plot")
        return

    choices = [
        "Regeneration BER and BSR vs threshold (multi-K)",
        "Regeneration BER vs threshold (multi-K)",
        "Regeneration BSR vs threshold (multi-K)",
        "Helper-equal BER vs threshold",
    ]

    selected = questionary.select(
        "Select a plotting configuration:", choices=choices
    ).ask()

    if selected == "Regeneration BER and BSR vs threshold (multi-K)":
        ks = [10, 100, 900]
        path = config.plot_global_ber_and_bsr_vs_threshold_multi(chips, ks)
        print(f"Saved: {path}")

    elif selected == "Regeneration BER vs threshold (multi-K)":
        ks = [10, 100, 900]
        path = config.plot_ber_vs_threshold_multi_K(chips, ks)
        print(f"Saved: {path}")

    elif selected == "Regeneration BSR vs threshold (multi-K)":
        ks = [10, 100, 900]
        path = config.plot_bsr_vs_threshold_multi_K(chips, ks, lambda_val=0.1)
        print(f"Saved: {path}")

    elif selected == "Helper-equal BER vs threshold":
        k_val = 100
        path = config.plot_helper_equal_vs_threshold(chips, k_val)
        print(f"Saved: {path}")

    else:
        print("No valid option selected.")


if __name__ == "__main__":
    main()


