""" Specialized helper data comparator for helperless stabilizer Bernardini.

This script implements the specific requirements mentioned in the user request:
- Reference range has parameters (K, delta) - fixed
- Tested ranges have parameters (N, D) - variable lists
- Processing is done using lists of N and D values
"""
import sys
import pathlib
from typing import List, Tuple, Dict
import numpy as np

# Add current directory to Python path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from common.data_reading_utils import get_files, read_readouts, ReadoutList
from ..evaluate_ber import build_bit_matrix, mask_from_threshold, majority_from_matrix
from .comparator_cache_manager import ComparatorCacheManager
from .helper_data_comparator import HelperDataComparator


class SpecializedHelperDataComparator:
    """ 
    Specialized comparator for helperless stabilizer Bernardini approach.
    
    This handles the specific case where:
    - Reference enrollment uses fixed parameters (K, delta)
    - Test enrollments use variable parameters (N, D) from lists
    """
    
    def __init__(self, readouts: ReadoutList, 
                 reference_K: int, reference_delta: float,
                 test_N_list: List[int], test_D_list: List[float]):
        """ Initialize the specialized comparator. """
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.reference_K = reference_K
        self.reference_delta = reference_delta
        self.test_N_list = test_N_list
        self.test_D_list = test_D_list
        self.cache_manager = ComparatorCacheManager()
        
        # Build bit matrix once for efficiency
        self.bit_matrix = build_bit_matrix(readouts)
        self.total_reads, self.num_cells = self.bit_matrix.shape

    def create_reference_mask_and_majority(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Create reference mask and majority using fixed (K, delta) parameters. """
        reference_mask = mask_from_threshold(self.bit_matrix, self.reference_K, self.reference_delta)
        reference_majority = majority_from_matrix(self.bit_matrix, self.reference_K)
        return reference_mask, reference_majority

    def create_test_mask_and_majority(self, N: int, D: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Create test mask and majority using variable (N, D) parameters. """
        test_mask = mask_from_threshold(self.bit_matrix, N, D)
        test_majority = majority_from_matrix(self.bit_matrix, N)
        return test_mask, test_majority


    def compare_masks(self, reference_mask: np.ndarray, test_mask: np.ndarray) -> Dict[str, float]:
        """ Compare reference and test masks and return statistics. """
        # Calculate mask disagreement (XOR operation)
        mask_disagreement = reference_mask ^ test_mask
        disagreement_rate = np.mean(mask_disagreement)
        
        # Calculate acceptance statistics
        reference_accepted = np.sum(reference_mask)
        test_accepted = np.sum(test_mask)
        
        # Calculate rates
        reference_acceptance_rate = reference_accepted / self.num_cells
        test_acceptance_rate = test_accepted / self.num_cells
        
        return {
            "disagreement_rate": disagreement_rate,
            "reference_acceptance_rate": reference_acceptance_rate,
            "test_acceptance_rate": test_acceptance_rate,
            "reference_accepted_count": reference_accepted,
            "test_accepted_count": test_accepted
        }

    def process_comparison_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """ Process comparison matrix for all (N, D) combinations. """
        print(f"Processing comparison matrix for chip {self.chip_id}")
        print(f"Reference parameters: K={self.reference_K}, delta={self.reference_delta}")
        print(f"Test N values: {self.test_N_list}")
        print(f"Test D values: {self.test_D_list}")
        
        # Create reference mask and majority
        reference_mask, _ = self.create_reference_mask_and_majority()
        
        # Initialize results dictionary
        results = {}
        
        # Process each (N, D) combination
        for N in self.test_N_list:
            results[N] = {}
            for D in self.test_D_list:
                print(f"  Processing N={N}, D={D}")
                
                # Create test mask and majority
                test_mask, _ = self.create_test_mask_and_majority(N, D)
                
                # Compare masks
                comparison_stats = self.compare_masks(reference_mask, test_mask)

                # Compute enrollment-only BER series vs K-th reference via core comparator
                ref_mask, ref_majority = reference_mask, majority_from_matrix(self.bit_matrix, self.reference_K)
                helper = HelperDataComparator(self.readouts, [float(D)], int(N), use_equal_ranges=False)
                err_series, cmp_series, rate_series = helper.compute_enrollment_ber_series(
                    ref_mask, ref_majority, int(N), float(D)
                )
                
                # Store results
                results[N][D] = {
                    **comparison_stats,
                    "enrollment_ber_error_counts": err_series.tolist(),
                    "enrollment_ber_compared_cells": cmp_series.tolist(),
                    "enrollment_ber_rates": rate_series.tolist(),
                }
        
        return results

    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                    output_file: str = None):
        """ Save results to file. """
        if output_file is None:
            output_file = f"specialized_comparison_{self.chip_id}_K{self.reference_K}_delta{self.reference_delta}.json"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")

    def print_summary(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """ Print summary of results. """
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Chip: {self.chip_id}")
        print(f"Reference: K={self.reference_K}, delta={self.reference_delta}")
        print(f"Total cells: {self.num_cells}")
        
        print("\nDisagreement Rates:")
        print("N\\D", end="")
        for D in self.test_D_list:
            print(f"\t{D:.3f}", end="")
        print()
        
        for N in self.test_N_list:
            print(f"{N}", end="")
            for D in self.test_D_list:
                disagreement = results[N][D]["disagreement_rate"]
                print(f"\t{disagreement:.4f}", end="")
            print()
        
        print("\nTest Acceptance Rates:")
        print("N\\D", end="")
        for D in self.test_D_list:
            print(f"\t{D:.3f}", end="")
        print()
        
        for N in self.test_N_list:
            print(f"{N}", end="")
            for D in self.test_D_list:
                acceptance = results[N][D]["test_acceptance_rate"]
                print(f"\t{acceptance:.4f}", end="")
            print()


def run_specialized_comparison(chip_ids: List[str] = None,
                              reference_K: int = 100, reference_delta: float = 0.2,
                              test_N_list: List[int] = None, test_D_list: List[float] = None):
    """ Run specialized comparison analysis. """
    if chip_ids is None:
        chip_ids = ['L45', 'M17', 'M39']  # Example chips
    
    if test_N_list is None:
        test_N_list = [50, 100, 150, 200]  # Example N values
    
    if test_D_list is None:
        test_D_list = [0.1, 0.15, 0.2, 0.25, 0.3]  # Example D values
    
    all_files = get_files()
    
    for chip_id in chip_ids:
        if chip_id not in all_files:
            print(f"Warning: Chip {chip_id} not found in data files")
            continue
        
        print(f"\nProcessing chip {chip_id}")
        readouts = read_readouts(all_files[chip_id])
        
        # Initialize specialized comparator
        comparator = SpecializedHelperDataComparator(
            readouts, reference_K, reference_delta, test_N_list, test_D_list
        )
        
        # Process comparison matrix
        results = comparator.process_comparison_matrix()
        
        # Print summary
        comparator.print_summary(results)
        
        # Save results
        comparator.save_results(results)


if __name__ == "__main__":
    print("Specialized Helper Data Comparator")
    print("=" * 60)
    
    # Example usage with different parameter combinations
    run_specialized_comparison(
        chip_ids=['L45', 'M17'],
        reference_K=100,
        reference_delta=0.2,
        test_N_list=[50, 100, 150, 200],
        test_D_list=[0.1, 0.15, 0.2, 0.25, 0.3]
    )
