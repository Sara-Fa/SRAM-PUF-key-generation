#!/usr/bin/env python3
"""
Plot histogram of one-probabilities (reliabilities) with theoretical PDF overlay.

This script generates a histogram showing the distribution of cell reliabilities
from real SRAM data, overlaid with the theoretical PDF f_Q(x; λ) from the
Bernardini-Rinaldo model.
"""

import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add plotting package directory to Python path for consistent relative imports
plot_root = pathlib.Path(__file__).parent / "plotting" / ".."
sys.path.insert(0, str(plot_root.resolve()))

from common.data_reading_utils import get_files, read_readouts, ReadoutList
from formulas import f_Q


def compute_cell_reliabilities(readouts: ReadoutList) -> np.ndarray:
    """
    Compute per-cell reliabilities from readouts.
    Returns array of shape (num_cells,) with reliability values.
    """
    if len(readouts) == 0:
        return np.array([])
    
    # Stack all readouts into a matrix
    bit_matrix = np.vstack([r.data.astype(np.uint8) for r in readouts])
    
    # Compute reliability as fraction of '1's for each cell
    reliabilities = np.mean(bit_matrix, axis=0).astype(np.float64)
    
    return reliabilities


def plot_reliability_histogram(chip_id: str, 
                             lambda_val: float,
                             reliabilities: np.ndarray,
                             output_dir: pathlib.Path,
                             num_bins: int = 20,
                             figsize: tuple = (6, 4.5)) -> None:
    """
    Plot histogram of reliabilities with theoretical PDF overlay.
    
    Args:
        chip_id: Chip identifier for title and filename
        lambda_val: Estimated lambda value for theoretical PDF
        reliabilities: Array of empirical reliability values
        output_dir: Directory to save the plot
        num_bins: Number of histogram bins
        figsize: Figure size (width, height)
    """
    
    # Create figure
    _, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    ax.hist(reliabilities, bins=num_bins, 
            density=True, alpha=0.7, 
            color='lightgray', edgecolor='black',
            linewidth=0.5, label='Empirical Data')
    
    # Create x values for theoretical PDF
    x_theory = np.linspace(0.001, 0.999, 1000)
    
    # Compute theoretical PDF f_Q(x; λ)
    pdf_theory = f_Q(x_theory, lambda_val)
    
    # Plot theoretical PDF
    ax.plot(x_theory, pdf_theory, 'r-', linewidth=3, 
            label=f'Theoretical PDF f_Q(x; λ={lambda_val:.3f})')
    
    # Set logarithmic y-axis
    ax.set_yscale('log')
    
    # Set axis labels and title
    ax.set_xlabel('x = one-probability (P_r)', fontsize=14)
    ax.set_ylabel('pdf_P_r (x)', fontsize=14)
    ax.set_title(f'Reliability Distribution - Chip {chip_id}\n'
                f'λ = {lambda_val:.6f}, N = {len(reliabilities):,} cells', 
                fontsize=14)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-2, 1e1)
    
    # Set x-axis ticks
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis ticks (log scale)
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels(['10⁻²', '10⁻¹', '10⁰', '10¹'])
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Add legend
    ax.legend(fontsize=12, frameon=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'reliability_histogram_{chip_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot: {output_file}")
    
    # Also save as PDF for vector graphics
    output_file_pdf = output_dir / f'reliability_histogram_{chip_id}.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved histogram plot (PDF): {output_file_pdf}")
    
    plt.show()


def plot_combined_histogram(lambda_json_path: pathlib.Path,
                           output_dir: pathlib.Path,
                           chip_ids: list = None,
                           num_bins: int = 20,
                           figsize: tuple = (7, 4),
                           lambda_for_plot: float = None) -> None:
    """
    Plot combined histogram of reliabilities across all chips with theoretical PDF overlay.
    
    Args:
        lambda_json_path: Path to lambda estimates JSON file
        output_dir: Directory to save the plot
        chip_ids: List of chip IDs to process (None for all)
        num_bins: Number of histogram bins
        figsize: Figure size (width, height)
        lambda_for_plot: Lambda value to use for theoretical PDF (None for average)
    """
    
    # Load lambda estimates
    with open(lambda_json_path, 'r', encoding='utf-8') as f:
        lambda_data = json.load(f)
    
    # Get all available files
    all_files = get_files()
    
    # Get available chip IDs
    available_chips = list(lambda_data.keys())
    if chip_ids is None:
        chip_ids = available_chips
    else:
        chip_ids = [cid for cid in chip_ids if cid in available_chips]
    
    print(f"Processing {len(chip_ids)} chips for combined histogram: {chip_ids}")
    
    # Collect all reliabilities across all chips
    all_reliabilities = []
    lambda_values = []
    chip_info = []
    
    for chip_id in chip_ids:
        print(f"  Processing chip {chip_id}...")
        
        # Get lambda value
        lambda_val = lambda_data[chip_id]['lambda_window']
        lambda_values.append(lambda_val)
        
        # Get file list for this chip
        if chip_id not in all_files:
            print(f"    Warning: No data files found for chip {chip_id}")
            continue
            
        file_list = all_files[chip_id]
        
        # Read all readouts for this chip
        readouts = read_readouts(file_list)
        
        # Compute reliabilities
        reliabilities = compute_cell_reliabilities(readouts)
        all_reliabilities.extend(reliabilities)
        cells_kb = len(reliabilities) / (8 * 1024)  # Convert to kB (8*1024 bits)
        chip_info.append(f"{chip_id} (λ={lambda_val:.3f}, N={cells_kb:.0f} kB)")
        
        print(f"    Added {len(reliabilities):,} cells ({cells_kb:.0f} kB)")
    
    all_reliabilities = np.array(all_reliabilities)
    total_cells = len(all_reliabilities)
    total_cells_kb = total_cells / (8 * 1024)  # Convert to kB (8*1024 bits)
    print(f"\nTotal cells across all chips: {total_cells:,} ({total_cells_kb:.0f} kB)")
    
    # Compute average lambda and print it
    if lambda_values:
        avg_lambda = np.mean(lambda_values)
        print(f"Average lambda across all chips: {avg_lambda:.6f}")
        print(f"Lambda range: {min(lambda_values):.6f} - {max(lambda_values):.6f}")
    
    # Determine which lambda to use for plotting
    if lambda_for_plot is None:
        plot_lambda = avg_lambda
        lambda_source = "average"
    else:
        plot_lambda = lambda_for_plot
        lambda_source = "user-specified"
    
    print(f"Using lambda = {plot_lambda:.6f} ({lambda_source}) for theoretical PDF")
    
    # Create figure
    _, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    ax.hist(all_reliabilities, bins=num_bins, 
            density=True, alpha=0.7, 
            color='lightgray', edgecolor='black',
            linewidth=0.5, label=f'Empirical')
    
    # Create x values for theoretical PDF
    x_theory = np.linspace(0.001, 0.999, 1000)
    
    # Compute theoretical PDF f_Q(x; λ) using specified lambda
    pdf_theory = f_Q(x_theory, plot_lambda)
    
    # Plot theoretical PDF
    ax.plot(x_theory, pdf_theory, 'r-', linewidth=3, 
            label=rf'Theoretical PDF $f_Q(p; λ=${plot_lambda:.1f}$)$')
    
    # Set logarithmic y-axis
    ax.set_yscale('log')
    
    # Set axis labels and title
    ax.set_xlabel('Probability of outputting \'1\' (p)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    # ax.set_title(f'Combined Reliability Distribution Across All Chips\n'
    #             f'λ_plot = {plot_lambda:.1f} ({lambda_source}), Total N = {total_cells_kb:.0f} kB', 
    #             fontsize=14)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-2, 1e1)
    
    # Set x-axis ticks
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis ticks (log scale)
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels(['10⁻²', '10⁻¹', '10⁰', '10¹'])
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Add legend
    ax.legend(fontsize=12, frameon=True)
    
    # # Add chip information as text
    # chip_text = '\n'.join(chip_info)
    # ax.text(0.02, 0.98, f"Chips included:\n{chip_text}", 
    #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # # Remove top and right spines for cleaner look
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'reliability_histogram_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined histogram plot: {output_file}")
    
    # Also save as PDF for vector graphics
    output_file_pdf = output_dir / 'reliability_histogram_combined.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved combined histogram plot (PDF): {output_file_pdf}")
    
    plt.show()


def plot_multiple_chips(lambda_json_path: pathlib.Path,
                       output_dir: pathlib.Path,
                       chip_ids: list = None,
                       num_bins: int = 20) -> None:
    """
    Plot reliability histograms for multiple chips.
    
    Args:
        lambda_json_path: Path to lambda estimates JSON file
        output_dir: Directory to save plots
        chip_ids: List of chip IDs to process (None for all)
        num_bins: Number of histogram bins
    """
    
    # Load lambda estimates
    with open(lambda_json_path, 'r', encoding='utf-8') as f:
        lambda_data = json.load(f)
    
    # Get all available files
    all_files = get_files()
    
    # Get available chip IDs
    available_chips = list(lambda_data.keys())
    if chip_ids is None:
        chip_ids = available_chips
    else:
        chip_ids = [cid for cid in chip_ids if cid in available_chips]
    
    print(f"Processing {len(chip_ids)} chips: {chip_ids}")
    
    # Process each chip
    for chip_id in chip_ids:
        print(f"\nProcessing chip {chip_id}...")
        
        # Get lambda value
        lambda_val = lambda_data[chip_id]['lambda_window']
        print(f"  λ = {lambda_val:.6f}")
        
        # Get file list for this chip
        if chip_id not in all_files:
            print(f"  Warning: No data files found for chip {chip_id}")
            continue
            
        file_list = all_files[chip_id]
        print(f"  Found {len(file_list)} readout files")
        
        # Read all readouts for this chip
        readouts = read_readouts(file_list)
        print(f"  Loaded {len(readouts)} readouts")
        
        # Compute reliabilities
        reliabilities = compute_cell_reliabilities(readouts)
        print(f"  Computed reliabilities for {len(reliabilities)} cells")
        
        # Plot histogram
        plot_reliability_histogram(chip_id, lambda_val, reliabilities, 
                                 output_dir, num_bins)


def main():
    """Main execution function."""
    
    # Set up paths
    script_dir = pathlib.Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "SRAM_readouts"
    lambda_json = (script_dir / "plotting" / ".." / "results" / "lambda_estimates.json").resolve()
    output_dir = (script_dir / "plotting" / ".." / "results" / "plots").resolve()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if lambda estimates exist
    if not lambda_json.exists():
        print(f"Error: Lambda estimates file not found: {lambda_json}")
        print("Please run lambda_estimation.py first.")
        return
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Configuration
    chip_ids_to_plot = None #['L45', 'M17', 'M39']  # Adjust as needed
    lambda_for_plot = 0.1  # Set the lambda value you want to use for plotting
    
    print("Generating reliability histograms with theoretical PDF overlay...")
    print(f"Data directory: {data_dir}")
    print(f"Lambda estimates: {lambda_json}")
    print(f"Output directory: {output_dir}")
    print(f"Lambda for plotting: {lambda_for_plot}")
    
    # Generate individual chip plots
    if chip_ids_to_plot is not None:
        print("\n=== Generating individual chip histograms ===")
        plot_multiple_chips(lambda_json, output_dir, 
                           chip_ids=chip_ids_to_plot, num_bins=20)
    
    # Generate combined histogram across all chips
    print("\n=== Generating combined histogram across all chips ===")
    plot_combined_histogram(lambda_json, output_dir, 
                           chip_ids=None, num_bins=20, lambda_for_plot=lambda_for_plot)


if __name__ == "__main__":
    main()
