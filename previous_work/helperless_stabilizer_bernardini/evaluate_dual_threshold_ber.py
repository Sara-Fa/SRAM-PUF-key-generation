#!/usr/bin/env python3
"""
Example script demonstrating dual-threshold BER evaluation.

This script shows how to use the parametric BER functions to:
1) Compute regeneration BER using threshold D on N enrollment reads
2) Compute enrollment BER between masks (delta,K) vs (D,N)

Outputs: CSV saved to `previous_work/helperless_stabilizer_bernardini/results/`.
"""

import pathlib
import sys
from typing import Dict, List

# Add current directory to Python path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from common.data_reading_utils import get_files, read_readouts
from evaluate_ber import (
    build_bit_matrix,
    evaluate_regen_ber_with_threshold,
    evaluate_enrollment_mask_ber
)


def evaluate_dual_threshold_ber(
    chip_ids: List[str] = None,
    K: int = 100,
    delta: float = 0.04,
    N: int = 200,
    D: float = 0.06
) -> Dict[str, Dict]:
    """
    Evaluate BER using dual thresholds: delta for base mask, D for regeneration.
    
    Args:
        chip_ids: List of chip IDs to process (None for all)
        K: Number of reads for base mask (delta threshold)
        delta: Threshold for base mask (delta)
        N: Number of reads for regeneration mask (D threshold)
        D: Threshold for regeneration mask (D > delta)
    
    Returns:
        Dictionary with results for each chip
    """
    if D <= delta:
        raise ValueError(f"D ({D}) must be greater than delta ({delta})")
    
    # Get all files
    all_files = get_files()
    if chip_ids is None:
        chip_ids = list(all_files.keys())
    
    results = {}
    
    for chip_id in chip_ids:
        if chip_id not in all_files:
            print(f"Warning: Chip {chip_id} not found, skipping")
            continue
            
        print(f"Processing chip {chip_id}...")
        
        # Load data
        file_list = all_files[chip_id]
        readouts = read_readouts(file_list)
        bit_matrix = build_bit_matrix(readouts)
        
        print(f"  Loaded {bit_matrix.shape[0]} reads, {bit_matrix.shape[1]} cells")
        
        # 1) Regeneration BER using D and N
        print(f"  Computing regeneration BER with D={D}, N={N}")
        accepted_frac, ber_mean, ber_min, ber_max, ber_var, heldout_rows = evaluate_regen_ber_with_threshold(
            bit_matrix=bit_matrix,
            N=N,
            threshold=D,
            heldout_start=None  # Use reads N to end
        )
        
        # 2) Enrollment mask BER: compare (delta,K) vs subsequent masks (D,N)
        print(f"  Computing enrollment mask BER: (delta={delta}, K={K}) vs (D={D}, N={N})")
        avg_mask_dis, blocks, cells = evaluate_enrollment_mask_ber(
            bit_matrix=bit_matrix,
            base_K=K,
            base_delta=delta,
            block_N=N,
            block_threshold=D,
            start_after_base=None  # Start after K reads
        )
        
        results[chip_id] = {
            'chip_id': chip_id,
            'total_reads': int(bit_matrix.shape[0]),
            'total_cells': int(bit_matrix.shape[1]),
            
            # Base mask parameters
            'base_K': K,
            'base_delta': delta,
            
            # Regeneration parameters
            'regen_N': N,
            'regen_D': D,
            
            # Regeneration BER results
            'regen_accepted_fraction': accepted_frac,
            'regen_ber_mean': ber_mean,
            'regen_ber_min': ber_min,
            'regen_ber_max': ber_max,
            'regen_heldout_rows': heldout_rows,
            
            # Enrollment mask BER results
            'enrollment_mask_disagreement': avg_mask_dis,
            'enrollment_blocks_used': blocks,
            'enrollment_cells_per_mask': cells,
        }
        
        print("  Results:")
        print(f"    Regeneration BER: {ber_mean:.6f} (accepted: {accepted_frac:.3f})")
        print(f"    Enrollment mask disagreement: {avg_mask_dis:.6f} ({blocks} blocks)")
        print()
    
    return results


def save_dual_threshold_results(results: Dict[str, Dict], output_dir: pathlib.Path) -> None:
    """Save dual-threshold results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'dual_threshold_ber_results.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('chip_id,total_reads,total_cells,base_K,base_delta,regen_N,regen_D,'
                'regen_accepted_fraction,regen_ber_mean,regen_ber_min,regen_ber_max,regen_heldout_rows,'
                'enrollment_mask_disagreement,enrollment_blocks_used,enrollment_cells_per_mask\n')
        
        for chip_data in results.values():
            f.write(f"{chip_data['chip_id']},{chip_data['total_reads']},{chip_data['total_cells']},"
                   f"{chip_data['base_K']},{chip_data['base_delta']},{chip_data['regen_N']},{chip_data['regen_D']},"
                   f"{chip_data['regen_accepted_fraction']},{chip_data['regen_ber_mean']},"
                   f"{chip_data['regen_ber_min']},{chip_data['regen_ber_max']},{chip_data['regen_heldout_rows']},"
                   f"{chip_data['enrollment_mask_disagreement']},{chip_data['enrollment_blocks_used']},"
                   f"{chip_data['enrollment_cells_per_mask']}\n")
    
    print(f"Saved dual-threshold results: {csv_file}")


def main():
    """Main execution function."""
    # Configuration parameters
    K = 500 #68          # Base mask: K reads with delta threshold
    delta = 0.499     # Base mask threshold (delta)
    N = 500 #68          # Regeneration mask: N reads with D threshold  
    D = 0.4991         # Regeneration mask threshold (D > delta)
    
    # Optional: specify specific chips to process
    chip_ids = None # ['L45'] # None for all chips, or e.g., ['L45', 'M17']
    
    print("Dual-Threshold BER Evaluation")
    print("=" * 50)
    print(f"Base mask: K={K}, delta={delta}")
    print(f"Regeneration mask: N={N}, D={D}")
    print(f"D > delta: {D} > {delta} = {D > delta}")
    print()
    
    # Evaluate BER
    results = evaluate_dual_threshold_ber(
        chip_ids=chip_ids,
        K=K,
        delta=delta,
        N=N,
        D=D
    )
    
    # Save results
    output_dir = pathlib.Path(__file__).parent / "plotting" / ".." / "results"
    output_dir = output_dir.resolve()
    save_dual_threshold_results(results, output_dir)

    # Summary
    print("Summary:")
    print(f"Processed {len(results)} chips")
    if results:
        avg_regen_ber = sum(r['regen_ber_mean'] for r in results.values() if not np.isnan(r['regen_ber_mean'])) / len(results)
        avg_enrollment_dis = sum(r['enrollment_mask_disagreement'] for r in results.values() if not np.isnan(r['enrollment_mask_disagreement'])) / len(results)
        print(f"Average regeneration BER: {avg_regen_ber:.6f}")
        print(f"Average enrollment mask disagreement: {avg_enrollment_dis:.6f}")


if __name__ == "__main__":
    import numpy as np
    main()
