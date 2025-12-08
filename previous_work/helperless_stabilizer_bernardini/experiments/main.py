"""Entry point for helperless stabilizer Bernardini experiments.

Wraps HelperDataComparator and GlobalBERProcessor to compute and cache
helper masks/BER per chip, then optionally report sample metrics.
"""
import sys
import pathlib
from typing import List

# Add current directory to Python path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from common.data_reading_utils import get_files, read_readouts
from .helper_data_comparator import HelperDataComparator
from .global_ber_processor import GlobalBERProcessor
from .analysis_utils import get_enrollment_threshold_values, get_enrollment_readings_values


def run_helper_data_comparison(chip_ids_list: List[str] = None, 
                             threshold_values: List[float] = None,
                             num_enroll_readings: int = 100):
    """ Run helper data comparison analysis. """
    print("=" * 60)
    print("HELPER DATA COMPARISON ANALYSIS")
    print("=" * 60)
    
    all_files = get_files()
    if chip_ids_list is None:
        chip_ids_list = list(all_files.keys())
    
    if threshold_values is None:
        threshold_values = get_enrollment_threshold_values()
    
    print(f"Processing chips: {chip_ids_list}")
    print(f"Threshold values: {threshold_values}")
    print(f"Enrollment readings: {num_enroll_readings}")
    
    for chip_id in chip_ids_list:
        if chip_id not in all_files:
            print(f"Warning: Chip {chip_id} not found in data files")
            continue
            
        print(f"\nProcessing chip {chip_id}")
        readouts = read_readouts(all_files[chip_id])
        
        # Initialize and run helper data comparator
        helper_comparator = HelperDataComparator(
            readouts, threshold_values, num_enroll_readings, use_equal_ranges=True
        )
        
        # Compute and save results
        helper_comparator.compare_and_save_helper_data()
        
        # Load and display results
        results = helper_comparator.initialize()
        print(f"Helper data comparison completed for chip {chip_id}")
        
        # Display sample results
        for threshold, rates in list(results.items())[:3]:  # Show first 3 thresholds
            print(f"  Threshold {threshold}:")
            print(f"    Error rate: {rates['error_rate'][0]:.6f}")
            print(f"    Acceptance rate: {rates['acceptance_rate'][0]:.6f}")
            print(f"    Extraction rate: {rates['extraction_rate'][0]:.6f}")


def run_global_ber_analysis(chip_ids_list: List[str] = None,
                          threshold_values: List[float] = None,
                          num_enroll_readings: int = 100,
                          process_all_ranges: bool = True,
                          iterative_ber: bool = True):
    """ Run global BER analysis. """
    print("=" * 60)
    print("GLOBAL BER ANALYSIS")
    print("=" * 60)
    
    all_files = get_files()
    if chip_ids_list is None:
        chip_ids_list = list(all_files.keys())
    
    if threshold_values is None:
        threshold_values = get_enrollment_threshold_values()
    
    print(f"Processing chips: {chip_ids_list}")
    print(f"Threshold values: {threshold_values}")
    print(f"Enrollment readings: {num_enroll_readings}")
    print(f"Process all ranges: {process_all_ranges}")
    print(f"Iterative BER: {iterative_ber}")
    
    for chip_id in chip_ids_list:
        if chip_id not in all_files:
            print(f"Warning: Chip {chip_id} not found in data files")
            continue
            
        print(f"\nProcessing chip {chip_id}")
        readouts = read_readouts(all_files[chip_id])
        
        # Initialize and run global BER processor
        ber_processor = GlobalBERProcessor(
            readouts, threshold_values, num_enroll_readings, process_all_ranges, iterative_ber
        )
        
        # Compute and save results
        ber_processor.compute_and_save_global_ber()
        
        # Load and display results
        results = ber_processor.initialize()
        print(f"Global BER analysis completed for chip {chip_id}")
        
        # Display sample results
        for threshold, rates in list(results.items())[:3]:  # Show first 3 thresholds
            print(f"  Threshold {threshold}:")
            print(f"    BER rate: {rates['ber_rate'][0]:.6f}")


def run_comparative_analysis(chip_ids_list: List[str] = None,
                            threshold_values: List[float] = None,
                            num_enroll_readings_list: List[int] = None):
    """ Run comparative analysis across different enrollment readings. """
    print("=" * 60)
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    all_files = get_files()
    if chip_ids_list is None:
        chip_ids_list = list(all_files.keys())
    
    if threshold_values is None:
        threshold_values = get_enrollment_threshold_values()
    
    if num_enroll_readings_list is None:
        num_enroll_readings_list = get_enrollment_readings_values()
    
    print(f"Processing chips: {chip_ids_list}")
    print(f"Threshold values: {threshold_values}")
    print(f"Enrollment readings: {num_enroll_readings_list}")
    
    for chip_id in chip_ids_list:
        if chip_id not in all_files:
            print(f"Warning: Chip {chip_id} not found in data files")
            continue
            
        print(f"\nProcessing chip {chip_id}")
        readouts = read_readouts(all_files[chip_id])
        
        # Run analysis for different enrollment readings
        for num_enroll_readings in num_enroll_readings_list:
            print(f"\n  Enrollment readings: {num_enroll_readings}")
            
            # Helper data comparison
            helper_comparator = HelperDataComparator(
                readouts, threshold_values, num_enroll_readings, use_equal_ranges=True
            )
            helper_comparator.compare_and_save_helper_data()
            
            # Global BER analysis
            ber_processor = GlobalBERProcessor(
                readouts, threshold_values, num_enroll_readings, process_all_ranges=True, iterative_ber=True
            )
            ber_processor.compute_and_save_global_ber()
            
            print(f"    Analysis completed for K={num_enroll_readings}")


if __name__ == "__main__":
    # Example usage
    print("Helperless Stabilizer Bernardini Experiments")
    print("=" * 60)
    
    # Define parameters
    chip_ids_list = ['L45', 'M17', 'M39']  # Example chips
    threshold_values = [0.1, 0.2, 0.3, 0.4]  # Example thresholds
    num_enroll_readings = 100  # Example enrollment readings
    
    # Run helper data comparison
    run_helper_data_comparison(chip_ids_list, threshold_values, num_enroll_readings)
    
    # Run global BER analysis - Option 1: Iterative BER (original behavior)
    print("\n" + "="*60)
    print("GLOBAL BER ANALYSIS - ITERATIVE BER COMPUTATION")
    print("="*60)
    run_global_ber_analysis(chip_ids_list, threshold_values, num_enroll_readings, 
                           process_all_ranges=True, iterative_ber=True)
    
    # Run global BER analysis - Option 2: Single BER computation (faster)
    print("\n" + "="*60)
    print("GLOBAL BER ANALYSIS - SINGLE BER COMPUTATION (FASTER)")
    print("="*60)
    run_global_ber_analysis(chip_ids_list, threshold_values, num_enroll_readings, 
                           process_all_ranges=True, iterative_ber=False)
    
    # Run comparative analysis (across different enrollment readings)
    num_enroll_readings_list = [50, 100, 150, 200]
    run_comparative_analysis(chip_ids_list, threshold_values, num_enroll_readings_list)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETED")
    print("=" * 60)
