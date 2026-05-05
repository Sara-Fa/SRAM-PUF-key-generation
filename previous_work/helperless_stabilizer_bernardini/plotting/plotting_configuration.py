"""Plotting configuration and functions.

Provides plotting functions for aggregated BER vs threshold and helper comparator.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import sys
import numpy as np
import pickle

# Ensure repository root is on sys.path for absolute imports when run as a script
repo_root = Path(__file__).parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from previous_work.helperless_stabilizer_bernardini.experiments.aggregate_results import (
    aggregate_global_ber_over_chips,
    aggregate_helper_equal_ranges_over_chips,
)
from previous_work.helperless_stabilizer_bernardini.experiments.global_ber_processor import GlobalBERProcessor
from common.data_reading_utils import get_files, read_readouts
from previous_work.helperless_stabilizer_bernardini.formulas import p_unreliable_exact


def _ensure_output_dir() -> Path:
    base = Path(__file__).parent.parent / "results" / "plots"
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_global_ber_and_bsr_vs_threshold_multi(
    chip_ids: List[str],
    num_enroll_readings_list: List[int],
) -> Path:
    out_dir = _ensure_output_dir()

    # Match p1: figure size (10,5), primary blue, secondary green, linestyles per K
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = 'blue'
    color2 = 'green'
    linestyles = ['dashed', 'dotted', 'solid', (0, (3, 1, 1, 1))]

    ax2 = ax1.twinx()

    # Plot BER (primary) and Acceptance (secondary) for each K with same linestyle
    for idx, K in enumerate(num_enroll_readings_list):
        thresholds, mean_ber, _std_ber, mean_accept, _std_accept = aggregate_global_ber_over_chips(chip_ids, K)
        if thresholds.size == 0:
            continue
        linestyle = linestyles[idx % len(linestyles)]
        ax1.plot(thresholds, mean_ber, linestyle=linestyle, color=color1, label=r'$K$='+f'{int(K)}')
        ax2.plot(thresholds, mean_accept, linestyle=linestyle, color=color2)

    # Axes styling like p1
    ax1.set_yscale("log")
    ax1.set_xlabel(r'Selection Threshold $\Delta$', fontsize=16, weight='bold')
    ax1.set_ylabel(r'$\mathbf{BER}_{\mathrm{Reg}}$', rotation=0, fontsize=16, color=color1, labelpad=20, weight='bold')
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(
        loc='upper left',
        bbox_to_anchor=(0, 0.8),
        fontsize=14,
        title='Nb. of Readings',
        title_fontsize=14,
    )
    ax1.yaxis.set_label_coords(-0.08, 0.6)

    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.set_ylabel('BSR', rotation=0, fontsize=16, color=color2, labelpad=20, weight='bold')

    ks_str = "_".join(str(int(k)) for k in num_enroll_readings_list)
    fname = out_dir / f"global_ber_vs_threshold_multi_K{ks_str}.pdf"
    fig.tight_layout()
    fig.savefig(fname)
    plt.show()
    return fname


def plot_ber_vs_threshold_multi_K(
    chip_ids: List[str],
    num_enroll_readings_list: List[int],
) -> Path:
    """Plot only BER vs threshold for multiple K values (single y-axis, log-scale)."""
    out_dir = _ensure_output_dir()

    fig, ax = plt.subplots(figsize= (8, 4))
    linestyles = ['dashed', 'dotted', 'solid', (0, (3, 1, 1, 1))]

    for idx, K in enumerate(num_enroll_readings_list):
        thresholds, mean_ber, _std_ber, _mean_accept, _std_accept = aggregate_global_ber_over_chips(chip_ids, K)
        print(f"At K={K}, thresholds={thresholds}, mean_ber={mean_ber}")
        if thresholds.size == 0:
            continue
        linestyle = linestyles[idx % len(linestyles)]
        ax.plot(thresholds, mean_ber, linestyle=linestyle, label=r'$K$='+f'{int(K)}')

    ax.set_yscale("log")
    ax.set_xlabel(r'Selection Threshold $\Delta$', fontsize=16, weight='bold')
    ax.set_ylabel(r'$\mathbf{BER}_{\mathrm{Reg}}$', rotation=0, labelpad=20, fontsize=16, weight='bold')
    ax.grid(True)
    ax.legend(title='Nb. of Readings', fontsize=14, title_fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_label_coords(-0.1, 0.6)

    ks_str = "_".join(str(int(k)) for k in num_enroll_readings_list)
    fname = out_dir / f"ber_vs_threshold_multi_ber_only_K{ks_str}.pdf"
    fig.tight_layout()
    fig.savefig(fname)
    plt.show()
    return fname


def plot_bsr_vs_threshold_multi_K(
    chip_ids: List[str],
    num_enroll_readings_list: List[int],
    lambda_val: float = 0.1,
) -> Path:
    """Plot only BSR vs threshold for multiple K values (single y-axis)."""
    out_dir = _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(8, 4))
    linestyles = ['dashed', 'dotted', 'solid', (0, (3, 1, 1, 1))]

    theory_thresholds = None
    theory_values = None
    empirical_lines = []

    for idx, K in enumerate(num_enroll_readings_list):
        thresholds, _mean_ber, _std_ber, mean_accept, _std_accept = aggregate_global_ber_over_chips(chip_ids, K)
        if thresholds.size == 0:
            continue
        linestyle = linestyles[idx % len(linestyles)]
        line, = ax.plot(
            thresholds,
            mean_accept,
            linestyle=linestyle,
            label=r'$K$='+f'{int(K)}'
        )
        empirical_lines.append(line)

        # Capture thresholds once to compute theoretical curve on same x-axis
        if theory_thresholds is None:
            theory_thresholds = thresholds

    # Add theoretical P[unrel] curve if thresholds are available
    theory_line = None
    if theory_thresholds is not None:
        theory_values = [1.0 - p_unreliable_exact(float(delta), float(lambda_val)) for delta in theory_thresholds]
        theory_line, = ax.plot(
            theory_thresholds,
            theory_values,
            color='red',
            linewidth=2.0,
            label='Theoretical 1 - P[unrel]'
        )

    ax.set_yscale("linear")
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel(r'Selection Threshold $\Delta$', fontsize=16, weight='bold')
    ax.set_ylabel('BSR', rotation=0, labelpad=20, fontsize=16, weight='bold')
    ax.grid(True)
    # Single vertically stacked legend at lower-left containing empirical curves and theory
    handles = list(empirical_lines)
    if theory_line is not None:
        handles.append(theory_line)
    if handles:
        ax.legend(handles=handles, loc='lower left', fontsize=14, ncol=1)
    ax.tick_params(axis='both', labelsize=12)

    ks_str = "_".join(str(int(k)) for k in num_enroll_readings_list)
    fname = out_dir / f"bsr_vs_threshold_multi_K{ks_str}.pdf"
    fig.tight_layout()
    fig.savefig(fname)
    plt.show()
    return fname

def plot_helper_equal_vs_threshold(
    chip_ids: List[str],
    K: int,
) -> Path:
    out_dir = _ensure_output_dir()
    thresholds, mean_ber, std_ber = aggregate_helper_equal_ranges_over_chips(chip_ids, K)
    if thresholds.size == 0:
        return out_dir / "helper_equal_vs_threshold_empty.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, mean_ber, marker="s", color="tab:blue", label="Mean BER")
    ax.fill_between(thresholds, mean_ber - std_ber, mean_ber + std_ber, alpha=0.2, color="tab:blue")
    ax.set_xlabel(r"Selection Threshold $\Delta$", fontweight="bold")
    ax.set_ylabel(r"$\\mathbf{BER}_{\\mathrm{Enr}}$", rotation=0, labelpad=20, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.tick_params(axis='both', labelsize=12)

    fname = out_dir / f"helper_equal_vs_threshold_K{int(K)}.pdf"
    fig.tight_layout()
    fig.savefig(fname)
    plt.show()
    return fname


def _load_bernardini_iterative_data_per_chip(chip_ids: List[str] = None, K: int = 900, threshold: float = 0.499, 
                                           cache_dir: Path = None):
    """
    Load iterative BER data per chip (not averaged over chips) for K=900 and delta=0.499.
    
    Args:
        chip_ids: List of chip IDs to process. If None, uses all available chips.
        K: Number of enrollment readings
        threshold: Threshold value
        cache_dir: Path to cache directory. If None, uses default relative to this file.
    
    Returns:
        dict: {chip_id: {'iterations': np.array, 'ber_per_iteration': np.array, 'ber_mean': np.array, 'ber_std': np.array}}
    """
    if cache_dir is None:
        # Default cache directory relative to this file's location
        cache_dir = Path(__file__).parent.parent / "experiments" / "cache"
    
    # Ensure cache directory exists
    cache_dir = Path(cache_dir)  # Convert to Path if it's a string
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_files = get_files()
    
    # If chip_ids is None, use all available chips
    if chip_ids is None:
        chip_ids = list(all_files.keys())
    
    per_chip_data = {}
    
    for chip_id in chip_ids:
        if chip_id not in all_files:
            continue
            
        # Load cache file for this chip and threshold
        threshold_str = f"{threshold:.3f}".replace('.', 'p')
        cache_file = cache_dir / f"regenerate_ber_{chip_id}_th{threshold_str}_num_readings{K}.pkl"
        
        if not cache_file.exists():
            print(f"Warning: Cache file not found for chip {chip_id}, threshold {threshold}, K={K}")
            print(f"  Expected file: {cache_file}")
            continue
            
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        except (pickle.PickleError, FileNotFoundError, PermissionError) as e:
            print(f"Error loading cache file for chip {chip_id}: {e}")
            continue
        
        error_count = np.asarray(data["error_count"])  # (ranges, K)
        valid_count = np.asarray(data["valid_patterns_count"])  # (ranges, K)
        
        # Validate data dimensions
        if error_count.ndim != 2 or valid_count.ndim != 2:
            print(f"Warning: Invalid data dimensions for chip {chip_id}. Expected 2D arrays.")
            continue
            
        if error_count.shape[1] != K or valid_count.shape[1] != K:
            print(f"Warning: Data length mismatch for chip {chip_id}. Expected K={K}, got {error_count.shape[1]}")
            continue
        
        # Get heldout count for this chip
        readouts = read_readouts(all_files[chip_id])
        total_reads = len(readouts)
        heldout_reads = max(total_reads - K, 0)
        num_cells = int(readouts[0].data.size) if total_reads > 0 else 1
        
        # Compute BER and acceptance rates for each iteration (1 to K)
        ber_rates, acc_rates = GlobalBERProcessor.get_rates_given_counts_single_threshold(
            (error_count, valid_count), heldout_reads, return_both=True, num_cells=num_cells
        )
        
        # Extract per-iteration data: ber_rates is (1, ranges, K), acc_rates is (1, ranges, K)
        ber_per_iteration = ber_rates[0, :, :]  # (ranges, K)
        acc_per_iteration = acc_rates[0, :, :]  # (ranges, K)
        
        # Average over ranges for each iteration
        ber_mean_per_iter = np.mean(ber_per_iteration, axis=0)  # (K,)
        ber_std_per_iter = np.std(ber_per_iteration, axis=0)   # (K,)
        acc_mean_per_iter = np.mean(acc_per_iteration, axis=0)  # (K,)
        acc_std_per_iter = np.std(acc_per_iteration, axis=0)   # (K,)
        
        iterations = np.arange(1, K + 1)
        
        per_chip_data[chip_id] = {
            'iterations': iterations,
            'ber_per_iteration': ber_per_iteration,  # (ranges, K)
            'ber_mean': ber_mean_per_iter,          # (K,)
            'ber_std': ber_std_per_iter,            # (K,)
            'acc_per_iteration': acc_per_iteration,  # (ranges, K)
            'acc_mean': acc_mean_per_iter,          # (K,)
            'acc_std': acc_std_per_iter,            # (K,)
        }
    
    return per_chip_data


